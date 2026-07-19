import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from absl import logging
from torch import nn
from transformers import GenerationConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (GenerateDecoderOnlyOutput,
                                           GenerateEncoderDecoderOutput,
                                           GenerateNonBeamOutput)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import ModelOutput, is_torchdynamo_compiling

from .logit_processor_3dim import (MultiModalLogitsProcessor_JACOBI,
                                   MultiTokensInterleavedTopKLogitsWarper,
                                   MultiTokensVLLogitsProcessor,
                                   gather_from_split_tensors,
                                   get_double_cfg_input_ids)


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def delete_false_key_value(
    self,
    delete_segments: list[tuple[int, int]],  
    device
) -> None:
    for layer_idx in range(len(self.key_cache)):
        seq_len = self.key_cache[layer_idx].shape[-2]  
        mask = torch.ones(seq_len, dtype=torch.bool, device=device) 
        for start, end in delete_segments:
            mask[start:end] = False
        valid_idx = mask.nonzero(as_tuple=True)[0] # 找到要保存的 token 索引
        self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim=-2, index=valid_idx)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim=-2, index=valid_idx)


def postprocess_cfg_decode(
    model_inputs,
    cfg_half_name_list=['inputs_embeds', 'input_ids', 'pixel_values', ],
):
    cfg_half_name_list = cfg_half_name_list
    def cfg_half(x):
        return x[:x.shape[0]//2]
    
    for name in cfg_half_name_list:
        if (name in model_inputs) and (model_inputs[name] is not None):
            model_inputs[name] = cfg_half(model_inputs[name])
    
    return model_inputs


def check_is_force_no_cfg(input_ids, image_start_token_id=None, image_end_token_id=None, guidance_scale=3.0, do_cfg=True):
    if (image_start_token_id is None) or (image_end_token_id is None):
        return False
    
    num_image_start_tokens = (input_ids[0] == image_start_token_id).sum()
    num_image_end_tokens = (input_ids[0] == image_end_token_id).sum()


def sampling_logits2tokens(
    logits,   # (batch_size, input_size, dim)
    all_collected_input_ids, # (batch_size, len)
    unfinished_sequences, pad_token_id, # (batch_size)
    output_token_num = 1, 
    logits_processor=None, logits_warper=None,
    do_sample=True,
    has_eos_stopping_criteria=True,
    do_cfg=False,
    guidance_scale=3.0,
    generator=None, #token_sampler = None,
    is_force_no_cfg = False,
):
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = logits[ :, -output_token_num:, : ].clone() #取最后几个token预测的概率

    if do_cfg:
        conditional_logits, unconditional_logits = next_token_logits.chunk(2, dim=0)
        if is_force_no_cfg:   # Text
            next_token_logits = conditional_logits
        else: # Image
            next_token_logits = guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits
    
    next_token_scores = logits_processor(all_collected_input_ids, next_token_logits) # 处理概率，约束

    if do_sample and (logits_warper is not None):
        next_token_scores = logits_warper(all_collected_input_ids, next_token_scores)

    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
        probs_shape = None 
        if len(probs.shape) >= 3:
            probs_shape = probs.shape
            probs = probs.flatten(0, len(probs_shape)-2) # (b*l, d)

        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1) # (b*l)
        if probs_shape is not None:
            next_tokens = next_tokens.reshape(probs_shape[:-1]) #(b*l) -> (b,l)
            probs = probs.reshape(probs_shape)  # (b*l, d) -> (b, l, d)

        next_token_scores = probs
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_token_scores = nn.functional.softmax(next_token_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if has_eos_stopping_criteria: # unfinished_sequences (B) [0, 1, ...0]. 0 means finished, add pad, otherwise add next_tokens
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    
    return next_tokens, next_token_scores


class SpeculativeSampler:

    def __init__(
        self, 
        collected_draft_logits=[], 
        collected_advanced_logits=[], 
        max_num_collected_logits=2,
        generator=None,
        draft_type = 'jacobian_states',
        reject_sampling_relative_ids = None,
        reject_sampling_draft_token_logits = None,
        sampling_last_draft_token = None,
    ):

        self.max_num_collected_logits = max_num_collected_logits #2
        self.collected_draft_logits = collected_draft_logits
        self.collected_advanced_logits = collected_advanced_logits

        self.draft_token_index_selector = lambda x: x
        if draft_type == 'jacobian_states':
            # for jacobi iteration (predict next token)
            self.advanced_token_index_selector = lambda x: x - 1
        else:
            self.advanced_token_index_selector = lambda x: x

        self.generator = generator

        self.image_token_list = [i for i in range(4, 8195 + 1)]

        self.reject_sampling_relative_ids = reject_sampling_relative_ids # [1]
        self.reject_sampling_draft_token_logits = reject_sampling_draft_token_logits # [1, 65536]
        self.sampling_last_draft_token = sampling_last_draft_token

        self._init_reject_sampling_params()
    
    def collect_logits(self, logits, collection_type='draft'):
        if collection_type == 'draft': 
            collected_logits = self.collected_draft_logits
        elif collection_type == 'advanced':
            collected_logits = self.collected_advanced_logits
        else:
            assert False, f"collection_type should be 'draft' or 'advanced', but got {collection_type}"

        if logits is not None:
            collected_logits.append(logits)
        
        if len(collected_logits) > self.max_num_collected_logits:
            return collected_logits.pop(0)
        else:
            return None
    
    def logits_calibrating(self, advanced_prob,):

        calibrated_logits = advanced_prob.log()

        B, L = advanced_prob.shape[:2]
        for b in range(B):
            reject_sampling_relative_index = self.reject_sampling_relative_ids[b]
            reject_sampling_draft_token_logits = self.reject_sampling_draft_token_logits[b]
            if reject_sampling_relative_index >= 0:
                token_advanced_prob = advanced_prob[b, reject_sampling_relative_index]

                calibrated_logits[b, reject_sampling_relative_index] = self.get_reject_sampling_logits(
                    token_advanced_prob, reject_sampling_draft_token_logits)
        
        self._init_reject_sampling_params()

        return calibrated_logits
    
    def get_reject_sampling_logits(self, token_advanced_prob, token_draft_prob):
        pos_delta_logits = (
            token_advanced_prob - token_draft_prob
        ).clamp(min=0).log()
        return pos_delta_logits
    
    def reject_sampling_single_token(
        self, token_advanced_prob,  #(d,)
        token_draft_prob, #(d,)
        logits_processor=None, logits_warper=None,
        all_collected_input_ids=None, #(l,)
    ):

        pos_delta_logits = self.get_reject_sampling_logits(token_advanced_prob, token_draft_prob) # (d)
        shape_pos_delta_logits = pos_delta_logits.shape 

        if (logits_processor is not None) or (logits_warper is not None):
            while len(all_collected_input_ids.shape) < 2: 
                all_collected_input_ids = all_collected_input_ids.unsqueeze(0) #(1,l)
            
            while len(pos_delta_logits.shape) < 3: 
                pos_delta_logits = pos_delta_logits.unsqueeze(0) #(1,1,d)
        
        if logits_processor is not None:
            pos_delta_logits = logits_processor(all_collected_input_ids, pos_delta_logits)
        
        # if logits_warper is not None:
        #     pos_delta_logits = logits_warper(all_collected_input_ids, pos_delta_logits) #(1,1,65536)

        pos_delta_logits = pos_delta_logits.view(shape_pos_delta_logits) # (65536)
        probs = F.softmax(pos_delta_logits, dim=-1)
        resampled_scores = probs # (V) 

        probs = probs.unsqueeze(0) if len(probs.shape) <= 1 else probs # (1, 65536)
       
        resampled_tokens = torch.multinomial(
            probs, num_samples=1, #len(probs.shape)-1,
            generator=self.generator,
        ).squeeze(-1)
        return resampled_tokens, resampled_scores 
        
    def _init_reject_sampling_params(self,):
        self.reject_sampling_relative_ids.fill_(-1)
        self.reject_sampling_draft_token_logits.fill_(0)
    
    def __call__(
        self, draft_tokens, # (b,l)
        advanced_tokens,  # (b,l)
        draft_prob, # (b,l,d)
        advanced_prob, # (b,l,d)
        logits_processor = None, logits_warper = None,
        **kwargs,
    ): 
        num_rows = kwargs.get("num_rows", None) # (k,) 
        acc_num_rows = torch.cumsum(num_rows, dim=0) # (2,2,3) => (2, 4, 7) [0,2) / [2,4) / [4,7)

        all_collected_input_ids = kwargs.get("all_collected_input_ids", None)
        # # reinitalize self.reject_sampling_relative_ids
        # self._init_reject_sampling_params()

        B, L = draft_tokens.shape
        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator) # (b,l,d)

        draft_token_index_selector = self.draft_token_index_selector # i
        advanced_token_index_selector = self.advanced_token_index_selector # i-1

        resampled_target_tokens = advanced_tokens.clone() # (b,l)
        resampled_target_scores = advanced_prob.clone() # (b,l,d)

        first_misaligned_token_inds_per_line = []  

        for b in range(B):
            batch_first_misaligned = [] 
            for row_idx in range(len(num_rows)):  # 0,1,2...,len(num_rows)-1  
                start_idx_in_row = acc_num_rows[row_idx - 1].item() if row_idx > 0 else 0
                end_idx_in_row = acc_num_rows[row_idx].item()
                first_misaligned_token_index = end_idx_in_row  
                #TODO 窗口大小为1，可以输入，生成未接收 token? 但是如果不输入呢
                if row_idx > 0 and start_idx_in_row+1==end_idx_in_row: 
                    first_misaligned_token_index = start_idx_in_row
                    batch_first_misaligned.append(first_misaligned_token_index)
                    continue
                elif row_idx > 0:
                    first_misaligned_token_index = end_idx_in_row - 1
                
                flag = True
                for i in range(start_idx_in_row+1, end_idx_in_row):
                    draft_token_index = draft_token_index_selector(i) #i
                    target_token_index = advanced_token_index_selector(i) #i-1
                    cls_idx = draft_tokens[b, draft_token_index]

                    sampled_adv = advanced_prob[b, target_token_index, cls_idx]
                    sampled_draft = draft_prob[b, draft_token_index, cls_idx]
                    r = rs[b, draft_token_index, cls_idx]

                    # self.sampling_last_draft_token[b] = cls_idx
                    if flag:
                        if r < (sampled_adv / sampled_draft).clamp(max=1):
                            resampled_target_tokens[b, target_token_index] = cls_idx
                            resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                        else:
                            if row_idx == 0:
                                first_misaligned_token_index = draft_token_index
                                resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                                    token_advanced_prob=advanced_prob[b, target_token_index, :],  # (dim)
                                    token_draft_prob=draft_prob[b, draft_token_index, :], # (dim)
                                    logits_processor=logits_processor,
                                    logits_warper=logits_warper,
                                    all_collected_input_ids=all_collected_input_ids, # (batch_size, seq_len)
                                )
                                resampled_target_tokens[b, target_token_index] = resampled_tokens
                                resampled_target_scores[b, target_token_index, :] = resampled_scores
                            else:
                                first_misaligned_token_index = draft_token_index - 1

                            flag = False
                    else:
                        if (sampled_adv / sampled_draft).clamp(max=1) > 0.5:
                            resampled_target_tokens[b, target_token_index] = cls_idx
                            resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]

                batch_first_misaligned.append(first_misaligned_token_index)

            first_misaligned_token_inds_per_line.append(batch_first_misaligned) # (b, k)

        return first_misaligned_token_inds_per_line, resampled_target_tokens, resampled_target_scores

def find_first_misaligned_token_inds(
    input_ids, next_tokens,
):
    first_misaligned_token_inds = []
    for b in range(input_ids.shape[0]):
        first_misaligned_token_index = input_ids.shape[1] #- 1 # keep at least one token left
        for i in range(1, input_ids.shape[1]):
            if input_ids[b, i] == next_tokens[b, i-1]:
                pass
            else:
                first_misaligned_token_index = i
                break
    
        first_misaligned_token_inds.append(first_misaligned_token_index)
    
    return first_misaligned_token_inds


def renew_pipeline(model_class): #FlexARInferenceSolver
    class JacobiPipeline(model_class): 
        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale  
            self.image_top_k = image_top_k 
            self.text_top_k = text_top_k 

        def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
            cfg = self.cfg if hasattr(self, 'cfg') else cfg
            image_top_k = self.image_top_k if hasattr(self, 'image_top_k') else image_top_k
            text_top_k = self.text_top_k if hasattr(self, 'text_top_k') else text_top_k

            logits_processor = LogitsProcessorList()

            candidate_processor = MultiModalLogitsProcessor_JACOBI( 
                image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
                image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
                patch_size=32,
                voc_size=self.model.config.vocab_size,
            )

            # candidate_processor = MultiTokensVLLogitsProcessor(
            #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token), #8197
            #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token), #8196
            #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token), #8803
            #     patch_size=32,
            #     voc_size=self.model.config.vocab_size, #65536
            #     device = self.device,
            # )

            topk_processor = MultiTokensInterleavedTopKLogitsWarper( # top-k
                image_top_k=image_top_k,
                text_top_k=text_top_k,
                image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            )

            logits_processor.append(candidate_processor)
            logits_processor.append(topk_processor)

            return logits_processor
    
    return JacobiPipeline

def get_multi_token_for_preparation( 
    img_vocab, 
    rand_token_num, # 每行需要初始化的token数目，配合ongoing_row_list可以得到总的初始化token数目
    ongoing_row_list,
    input_ids, temporary_collected_scores, device,   # 已有概率分布
):
    # NOTE 计算所有行的总jacobi token数
    rand_token_num = rand_token_num[ongoing_row_list]
    rand_token_num = rand_token_num.sum(dim=-1) 

    # NOTE 随机化对应个数的jacobi token和概率 数目取值[0, num] 
    img_vocab = img_vocab.to(device) # codebook
    img_vocab_size = len(img_vocab) 
    rand_tokens = torch.randint(  
        0, img_vocab_size,   # [0~8191]随机取数
        (*input_ids.shape[:-1], rand_token_num) 
    ).to(device)
    scores_of_rand_tokens = temporary_collected_scores.new_zeros( 
        (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1]) 
    )
    scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)
 
    return rand_tokens, scores_of_rand_tokens 

def renew_sampler(model_class): #ChameleonForConditionalImageGeneration
    
    class JacobiSampler(model_class, nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()

        def prepare_inputs_for_generation_jacobi( 
            self,
            input_ids,
            pixel_values=None,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,   
            use_cache=True,
            **kwargs,
        ):  
            # filling random tokens for multi-next-token prediction
            batch_size, all_collected_length = input_ids.shape

            is_append_random_tokens = kwargs.get("is_append_random_tokens", False) # True
            additional_tokens = kwargs.get("additional_tokens", None) #None  if output_num=1,this vari is always none
            additional_scores = kwargs.get("additional_scores", None) #None
            temporary_collected_scores = kwargs.get("temporary_collected_scores", None) 

            # NOTE 1.随机初始化每行的jacobi token
            if is_append_random_tokens:     
                rand_tokens, scores_of_rand_tokens = get_multi_token_for_preparation(
                    img_vocab=self.img_vocab,
                    rand_token_num=self.jacobi_init_num,   
                    ongoing_row_list=self.ongoing_row_list,
                    input_ids=input_ids, 
                    temporary_collected_scores = temporary_collected_scores,
                    device = input_ids.device,
                )
            
                # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
                # Exception 1: when passing input_embeds, input_ids may be missing entries
                # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
                # NOTE 2.构造输入，每行的输入由每行最后一个token，上一轮unmatched_token和jacobi token组成。 计算position_ids, kv cache position
                seq_len = 0 # 输入序列长度
                target_len = past_key_values.get_seq_length() if past_key_values is not None else 0 # kv cache长度+输入序列长度， 初始化设置为已生成序列长度，加上总的jacobi_num即得到最终长度
              
                if past_key_values is not None:
                    new_input_ids = [] # (b, l)  draft_tokens
                    new_input_scores = [] # draft_scores

                    new_position_ids = [] ## position_ids = global
                    local_position_ids = [] ## kv cache position

                    if additional_tokens is not None:
                        additional_num_rows = self.unmatched_num[self.ongoing_row_list] # (K,) 有没有可能出现上一行的token全部生成完毕，但是unmatched_num > 0? 不会，假设i行全部生成完毕，那么这一轮的输入肯定都验证通过，验证通过表示unmatched=0，因为如果有没验证通过的，那么这一行肯定没有生成完毕
                        additional_num_indices = additional_num_rows[:-1].cumsum(0) 

                        additional_token_rows = torch.tensor_split(additional_tokens, additional_num_indices, dim=1)  # len(self.ongoing_row_list)
                        additional_token_score_rows = torch.tensor_split(additional_scores, additional_num_indices, dim=1)

                    # 2.将jacobi内的token划分成多行
                    jacobi_num_rows = self.jacobi_init_num[self.ongoing_row_list] # (k,)
                    jacobi_init_indices = jacobi_num_rows[:-1].cumsum(0)

                    jacobi_token_rows = torch.tensor_split(rand_tokens, jacobi_init_indices, dim=1) #len(self.ongoing_row_list)
                    jacobi_token_score_rows = torch.tensor_split(scores_of_rand_tokens, jacobi_init_indices, dim=1)

                    cur_loc = 0
                    if kwargs['last_token_in_row_idx'] is not None: # 上一行的末尾
                        idx_in_input_ids = kwargs['last_token_in_row_idx']
                        # x = self.prompt_len + 2 + torch.sum(self.row_token_num[:self.ongoing_row_list[0]], dim=0) - 1
                        # print(self.row_token_num)
                        # print(idx_in_input_ids)
                        # print(x)
                        assert idx_in_input_ids == (self.prompt_len + 2 
                                                    + torch.sum(self.row_token_num[:self.ongoing_row_list[0]], dim=0) - 1
                                                    )
                        global_idx = (
                                        self.prompt_len + 2 
                                        + (self.ongoing_row_list[0]) * self.tokens_per_row -1 
                                    )
                        
                        new_input_ids.append(input_ids[:, idx_in_input_ids].unsqueeze(-1)) # (batch_size, 1)
                        new_input_scores.append(temporary_collected_scores[:, idx_in_input_ids].unsqueeze(1)) # (batch_size, 1, dim)

                        new_position_ids.append(global_idx)   #new_position收集的是全局索引
                        local_position_ids.append(idx_in_input_ids)   #local_position收集的是token的局部索引
                        
                        seq_len += 1
                        target_len += 1
                        cur_loc += 1

                    # print("inputs_id len:", input_ids.shape[-1])
                    # print("row_token_num:", self.row_token_num)
                    # print("keep_unmatched_num:", self.keep_unmatched_num)
                    # print("jacobi_num:", self.jacobi_init_num)
                    # print("verify_num:", self.verify_num)
                    for i in range(len(self.ongoing_row_list)): # 每一行处理
                        row = self.ongoing_row_list[i]  # 行索引
                        if self.verify_num[row] == 0: 
                            continue
                        idx_start_in_input_ids = (
                                                    self.prompt_len + 2 
                                                    + torch.sum(self.row_token_num[:(row + 1)], dim=0) 
                                                    - 1
                                                )       # 每行第一个 0,1,2..,row
                        
                        new_input_ids.append(input_ids[:, idx_start_in_input_ids].unsqueeze(-1))
                        new_input_scores.append(temporary_collected_scores[:, idx_start_in_input_ids].unsqueeze(1))
                        
                        jacobi_num = self.verify_num[row] - 1
                        idx_in_input_ids = (
                                            self.prompt_len + 2
                                            + torch.sum(self.row_token_num[:(row + 1)], dim=0) # 已生成
                                            + torch.sum(self.keep_unmatched_num[:row], dim=0) # 未匹配
                                            + torch.sum(self.jacobi_init_num[:row], dim=0) - 1 # 随机初始化
                                            )
                        local_position_ids.extend(range(idx_in_input_ids, idx_in_input_ids+jacobi_num+1))

                        global_idx = (
                                        self.prompt_len + 2 
                                        + row * self.tokens_per_row 
                                        + self.row_token_num[row] - 1
                                    )
                        new_position_ids.extend(range(global_idx, global_idx+jacobi_num+1)) #[global_idx, global_idx+1, ... global_idx+jacobi_num]
                        
                        self.start_location_ids[row] = idx_in_input_ids + 1 # 用于构造 attention_mask (jacobi第二个，排除第一个，第一个是已经接收的)
                        self.end_location_ids[row] = idx_in_input_ids + 1 + jacobi_num  
                        self.cur_start_location_ids[row] = cur_loc

                        seq_len += self.verify_num[row]
                        target_len += self.verify_num[row]
                        cur_loc += seq_len

                        #TODO add additional_tokens, add init_token
                        if additional_tokens is not None:
                            new_input_ids.append(additional_token_rows[i][:, :self.keep_unmatched_num[row]]) # 未匹配的 unmatched
                            new_input_scores.append(additional_token_score_rows[i][:, :self.keep_unmatched_num[row]]) 
                        
                        new_input_ids.append(jacobi_token_rows[i]) # jacobi 初始化的 jacobi_init
                        new_input_scores.append(jacobi_token_score_rows[i])

                    
                    input_ids = torch.cat(new_input_ids, dim=1)
                    input_token_scores = torch.cat(new_input_scores, dim=1) 
                    position_ids = torch.tensor(new_position_ids, device=input_ids.device).unsqueeze_(0) # 用于计算位置编码 (1, seq_len)
                    local_position_ids = torch.tensor(local_position_ids, device=input_ids.device).unsqueeze_(0) # 用于kv cache
                    # print(local_position_ids)

                # 准备position_ids
                position_ids = position_ids.repeat(batch_size*2, 1) # (batch_size*2, seq_len)
                position_ids[batch_size:, :] = position_ids[batch_size:, :] - self.prompt_len

                # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
                if inputs_embeds is not None and cache_position[0] == 0:
                    model_inputs = {"inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

                if cache_position[0] == 0:
                    # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
                    # Otherwise we need pixel values to be passed to model
                    model_inputs["pixel_values"] = pixel_values

                # 准备 attention_mask
                dtype, device = input_ids.dtype, input_ids.device
                attention_mask = torch.ones((batch_size*2, seq_len, target_len), dtype=dtype, device=device)

                for r in self.ongoing_row_list:
                    r_start_id = self.cur_start_location_ids[r] # 这一行 token 在输入序列中的相对位置
                    verify_num = self.verify_num[r]

                    for less_r in self.ongoing_row_list:
                        if less_r < r:
                            c_start_id = self.start_location_ids[less_r] # 之前行待验证窗口的起始位置
                            c_end_id = self.end_location_ids[less_r]
                            attention_mask[:, r_start_id:r_start_id+verify_num, c_start_id:c_end_id] = 0
                        else:
                            break
                attention_mask[batch_size:, :, :self.prompt_len-1] = 0   # 保证 cfg 无条件
                
                if local_position_ids.shape[1] != input_ids.shape[1]:
                    pass
                model_inputs.update(
                    {
                        "position_ids": position_ids,
                        "cache_position": cache_position,
                        "past_key_values": past_key_values,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "local_position_ids": local_position_ids,
                        'input_token_scores': input_token_scores,
                    }
                )

            return model_inputs

        def prepare_inputs_for_generation(
            self, 
            input_ids,
            input_scores,
            pixel_values=None,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,):
            if past_key_values is not None:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]
                    input_scores = input_scores[:, cache_position]

            if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1 # (batch_size, seq_len) or (batch_size, seq_len, target_len)
                    position_ids.masked_fill_(attention_mask == 0, 1) # 把无效 token 的位置设置为 1

                    if len(position_ids.shape) == 3: ###!!!
                        position_ids = position_ids[:, -1, :] # (batch_size, target_len)
                    
                    if past_key_values:
                        position_ids = position_ids[:, -input_ids.shape[1] :] # (batch_size, seq_len)

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

            if cache_position[0] == 0:
                # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
                # Otherwise we need pixel values to be passed to model
                model_inputs["pixel_values"] = pixel_values

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                    "input_token_scores": input_scores,
                }
            )
            return model_inputs

        def position_insert(self, input_ids, next_tokens, position): # 将next_tokens插入position
            if isinstance(position, torch.Tensor):
                position = position.item()
            # if len(input_ids.shape) == 2:
            #     if next_tokens.shape[1] > 0:
            #         mask = (next_tokens == 8803)
            #         exists = mask.any()
            #         if exists:
            #             pass
            return torch.cat((input_ids[:, :position], next_tokens, input_ids[:, position:]), dim=1)
    
        def prefix_matching_next_tokens(
            self,
            model_input_ids,  
            next_tokens, 
            next_token_scores, 
            input_token_scores = None, 
            prefix_token_sampler=None,
            **kwargs,
        ):  
            if len(self.ongoing_row_list)==0: 
                matched_num = model_input_ids.shape[1] 
                unmatched_tokens_num = 0

                matched_next_tokens = next_tokens[:, -1:] 
                unmatched_next_tokens = next_tokens[:, next_tokens.shape[1]:] 

                matched_next_scores = next_token_scores[:, -1:]
                unmatched_next_scores = next_token_scores[:, next_token_scores.shape[1]:] 

                min_accept_counts = None
            else:
                num_rows = self.verify_num[self.ongoing_row_list] 
                acc_num_rows = torch.cumsum(num_rows, dim=0) 

                kwargs['num_rows'] = num_rows
                if prefix_token_sampler is not None:
                    first_misaligned_input_token_inds, next_tokens, next_token_scores = prefix_token_sampler(  
                        draft_tokens = model_input_ids,  
                        advanced_tokens = next_tokens, 
                        draft_prob = input_token_scores, 
                        advanced_prob = next_token_scores,
                        **kwargs,
                    )
                    # first_misaligned_input_token_inds: [batch_size, lines]   每一个元素标记第一个拒绝的元素位置
                    first_misaligned_input_token_inds_tensors = torch.tensor(first_misaligned_input_token_inds)  # [batch_size, lines]

                    # (lines, ) 标记每一行起始的元素位置
                    row_starts = torch.cat([torch.tensor([0], device=acc_num_rows.device), acc_num_rows[:-1]]) # [0, 2, 4]
                    row_starts = row_starts.unsqueeze(0)  # [1, lines]

                    # (batch_size, lines) 标记每一行接收的token
                    relative_accept_counts = first_misaligned_input_token_inds_tensors - row_starts  
                    min_accept_counts = relative_accept_counts.min(dim=0).values  #
                    unmatched_tokens_num = num_rows - min_accept_counts # (batch_size, lines)

                matched_next_tokens_per_row = []
                unmatched_next_tokens_per_row = []
                matched_next_scores_per_row = []
                unmatched_next_scores_per_row = []

                for row_idx in range(len(num_rows)):  
                    row_start = row_starts[0, row_idx].item() # 这一行在序列中的起始位置
                    row_end = acc_num_rows[row_idx].item() # 这一行在序列中的结束为止
                    matched_len = min_accept_counts[row_idx].item() # 这一行通过验证的token数

                    matched_next_tokens_per_row.append(next_tokens[:, row_start:row_start+matched_len]) # 把这一行匹配的结果保存到数组
                    unmatched_next_tokens_per_row.append(next_tokens[:, row_start+matched_len:row_end]) # 把这一行未匹配的结果保存到数组

                    matched_next_scores_per_row.append(next_token_scores[:, row_start:row_start+matched_len])
                    unmatched_next_scores_per_row.append(next_token_scores[:, row_start+matched_len:row_end])

              
                matched_next_tokens = torch.cat(matched_next_tokens_per_row, dim=1)   # [batch_size, sum(matched_len)]
                unmatched_next_tokens = torch.cat(unmatched_next_tokens_per_row, dim=1)
                matched_next_scores = torch.cat(matched_next_scores_per_row, dim=1)
                unmatched_next_scores = torch.cat(unmatched_next_scores_per_row, dim=1)

            return min_accept_counts, unmatched_tokens_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores
        
        def prepare_cfg_input(
            self, 
            model_inputs, 
            cfg_repeat_name_list, 
            prefill_num=None,
            neg_input_ids = None,
        ):
            def cfg_repeat(x):
                return x.repeat(2, *([1] * (len(x.shape) - 1)))

            for name in cfg_repeat_name_list:
                if (name in model_inputs) and (model_inputs[name] is not None):
                    if name == 'attention_mask':   # (batch_size, len)->(batch_size*2, len)
                        model_inputs[name] = cfg_repeat(model_inputs[name])
                        B = model_inputs[name].shape[0]
                        model_inputs[name][B//2:, :prefill_num] = 0 
                    elif name == 'input_ids' and neg_input_ids is not None:
                        input_ids = model_inputs[name]
                        neg_input_ids = neg_input_ids
                        model_inputs[name] = get_double_cfg_input_ids(
                            input_ids, 
                            neg_input_ids,
                            pad_category = self.config.pad_token_id,
                        )
                    else:
                        model_inputs[name] = cfg_repeat(model_inputs[name])
             
            return model_inputs

        def _get_initial_cache_position(self, input_ids, model_kwargs):
            """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
            # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
            if "inputs_embeds" in model_kwargs:
                cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            else:
                cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1 # (seq_len,)

            past_length = 0
            if model_kwargs.get("past_key_values") is not None:
                cache = model_kwargs["past_key_values"]
                past_length = 0
                if not isinstance(cache, Cache):
                    past_length = cache[0][0].shape[2]
                elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                    past_length = cache.get_seq_length() # 0

                # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
                # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
                if not is_torchdynamo_compiling():
                    cache_position = cache_position[past_length:]

            model_kwargs["cache_position"] = cache_position

            return model_kwargs
        
        def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:  #更新past_key_values和attention_mask
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            model_kwargs[cache_name] = cache   #保存最新的cache
            if getattr(outputs, "state", None) is not None:
                model_kwargs["state"] = outputs.state

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            if not is_encoder_decoder: #decoder_only
                # update attention mask
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]

                    while len(attention_mask.shape) < 3:  #after one step, transform (b,l) to (b,1,l)  
                        attention_mask = attention_mask.unsqueeze(1)
                    
                    attention_mask = attention_mask[..., -1:, :] 
                    #TODO 
                    model_kwargs["attention_mask"] = torch.ones( 
                        ( attention_mask.shape[0], num_new_tokens, num_new_tokens + attention_mask.shape[-1]), #broadcast
                        device=attention_mask.device, dtype=attention_mask.dtype,
                    )
               
                    model_kwargs["attention_mask"][..., :, :attention_mask.shape[-1]] = attention_mask[..., -1:, :] #
                    model_kwargs["attention_mask"][..., :, attention_mask.shape[-1]:] = torch.tril(
                        model_kwargs["attention_mask"][0, :, attention_mask.shape[-1]:]
                    )
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )

            if model_kwargs.get("use_cache", True): #update cache_position
                # if num_new_tokens <= 1:
                #     model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = new_positions
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            
            return model_kwargs

        def _init_new_params(
            self, 
            max_num_new_tokens = 16,
            guidance_scale = 3.0,
            seed = 42,
            multi_token_init_scheme = 'random',
            do_cfg = True,
            prefix_token_sampler_scheme = 'speculative_jacobi',
            use_chameleon_tokenizer = True,
            _init_doubled_attn_mask_cfg = False,
            target_size = 768,
            **kwargs,
        ):
            if use_chameleon_tokenizer:
                import model.chameleon_vae_ori as chameleon_vae_ori
                chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
                    json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json"))["model"]["vocab"]
                )
                chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(chameleon_ori_vocab)
                img_vocab = chameleon_ori_translation._vocab.image_tokens
                self.register_buffer("img_vocab", torch.tensor(img_vocab, dtype=torch.long))
            else:
                if not hasattr(self, 'img_vocab'):
                    self.img_vocab = None

            self.cfg_repeat_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 
            ]
            self.cfg_half_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 
            ]

            # self.max_num_new_tokens = max_num_new_tokens
            # self.max_jacobi_iter_num = min(200, self.max_num_new_tokens+1) ###!!!
            self.guidance_scale = guidance_scale

            self.seed = seed
            self.generator = None

            self.multi_token_init_scheme = multi_token_init_scheme
            self.do_cfg = do_cfg

            self.prefix_token_sampler_scheme = prefix_token_sampler_scheme
            self._init_doubled_attn_mask_cfg = _init_doubled_attn_mask_cfg

            self.h_latent_dim = target_size // 16
            self.w_latent_dim = target_size // 16

            self.tokens_per_row = self.w_latent_dim + 1 #加上行末尾符号

            self.eol_token = torch.tensor([8803], device='cuda')
            self.eoi_token = torch.tensor([8196], device='cuda')
            self.end_token = torch.tensor([8710], device='cuda')
            self.pad_token = torch.tensor([1], device='cuda')
            
            self.window_size = 8
            self.jacobi_size = 1
            self.long_size = 9

        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,    # False
            streamer,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                logits_warper (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                    to warp the prediction score distribution of the language modeling head applied before multinomial
                    sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                    `generation_config`)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor 
            # assert False, f"pad_token_id: {pad_token_id}"
            output_attentions = generation_config.output_attentions 
            output_hidden_states = generation_config.output_hidden_states 
            output_scores = generation_config.output_scores 
            output_logits = generation_config.output_logits 
            return_dict_in_generate = generation_config.return_dict_in_generate 
            max_length = generation_config.max_length 
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria) # true
            do_sample = generation_config.do_sample # true
            # if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            #     raise ValueError(
            #         "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            #         f"{logits_warper})."
            #     )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None #none
            raw_logits = () if (return_dict_in_generate and output_logits) else None #none
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None #none
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None #none
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None #none

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            self.ongoing_row_list = []

            self.row_token_num = torch.zeros((self.tokens_per_row-1,), dtype=torch.long) # generated
            self.unmatched_num = torch.zeros((self.tokens_per_row-1,), dtype=torch.long) 
            self.keep_unmatched_num = torch.zeros((self.tokens_per_row-1,), dtype=torch.long) 
            self.jacobi_init_num = torch.zeros((self.tokens_per_row-1,), dtype = torch.long) # num-keep_unmatched-1
            self.verify_num = torch.zeros((self.tokens_per_row-1,), dtype=torch.long) # keep_unmatched+jacobi+1

            self.start_location_ids = torch.zeros((self.tokens_per_row-1,), dtype=torch.long)
            self.end_location_ids = torch.zeros((self.tokens_per_row-1,), dtype=torch.long)
            self.cur_start_location_ids = torch.zeros((self.tokens_per_row-1,), dtype=torch.long)

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape  
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device) # (batch_size,)
          
            temporary_collected_scores = input_ids.new_zeros((batch_size, cur_len, self.config.vocab_size)) # (batch_size, len, dim)
            temporary_collected_scores = torch.scatter(temporary_collected_scores, 2, input_ids.unsqueeze(-1), 1.0) #

            # init: cache_position 
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs) 
            model_kwargs['last_token_in_row_idx'] = None ## special case, we need to input last token in row to model to generate KV cache

            self.prompt_len = cur_len

            do_cfg = self.do_cfg if hasattr(self, 'do_cfg') else False
            guidance_scale = self.guidance_scale if hasattr(self, 'guidance_scale') else 3.0 
            do_cfg = (do_cfg & (guidance_scale != 1)) 
            if do_cfg: 
                model_kwargs = self.prepare_cfg_input(
                    model_kwargs, 
                    cfg_repeat_name_list = ['attention_mask', ] if (
                        not self._init_doubled_attn_mask_cfg
                    ) else [],
                    prefill_num = self.prompt_len - 1 ,
                )

            additional_tokens = None
            additional_scores = None

            if self.seed is not None:
                set_seed(self.seed)
                self.generator = torch.Generator(input_ids.device).manual_seed(self.seed)

            gen_loop_num = 0
            device = input_ids.device
            dtype = input_ids.dtype

            if self.prefix_token_sampler_scheme == 'speculative_jacobi':
                prefix_token_sampler = SpeculativeSampler(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        (batch_size, ), dtype=dtype, device=device
                    ),
                )
            elif self.prefix_token_sampler_scheme == 'jacobi':
                prefix_token_sampler = None
            else:
                raise ValueError(f"prefix_token_sampler_scheme: {self.prefix_token_sampler_scheme}")

            count_time = True
            if count_time:
                t1 = torch.cuda.Event(enable_timing=True)
                t2 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t1.record()

            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=device, cur_len=cur_len, max_length=max_length
            ):
                # prepare model inputs
                num_image_start_tokens = (input_ids[0] == 8197).sum() 
                if num_image_start_tokens >= 1: 
                    image_start_token_id_index = torch.where(input_ids[0] == 8197)[0][-1].item() 
                    image_token_num = len(input_ids[0][image_start_token_id_index + 1 :]) 
                else: 
                    image_token_num = 0
                
                if image_token_num == 2: 
                    self.ongoing_row_list.append(0) 
                    self.row_token_num[0] += 1 
                    self.verify_num[0] += 1
                   
                if image_token_num > 2: 
                    model_inputs = self.prepare_inputs_for_generation_jacobi(
                        input_ids, # (batch_size, seq_len)
                        is_append_random_tokens = True,
                        additional_tokens = additional_tokens,
                        additional_scores = additional_scores,
                        temporary_collected_scores = temporary_collected_scores,
                        img_width = logits_processor[0].w_latent_dim if hasattr(logits_processor[0], 'w_latent_dim') else None,
                        **model_kwargs,
                    )
                else: 
                    model_inputs = self.prepare_inputs_for_generation(input_ids, temporary_collected_scores, **model_kwargs)

                # num_new_tokens = model_inputs['input_ids'].shape[1] if len(self.ongoing_row_list) > 0 else 1

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # the first element of model_inputs['input_ids'] is in all_collected_input_ids
                all_collected_input_ids = input_ids 
                model_input_ids = model_inputs['input_ids'] # input

                input_token_scores = model_inputs.pop('input_token_scores')
                # input_ids_accept_len_ptr = model_inputs.pop('input_ids_accept_len_ptr')
                # rand_tokens_shape = model_inputs.pop('rand_tokens_shape')

                is_force_no_cfg = check_is_force_no_cfg( 
                    input_ids, 
                    image_start_token_id = logits_processor[0].image_start_token_id if hasattr(
                        logits_processor[0], 'image_start_token_id'
                    ) else None,
                    image_end_token_id = logits_processor[0].image_end_token_id if hasattr(
                        logits_processor[0], 'image_end_token_id'
                    ) else None,
                    guidance_scale=guidance_scale,
                    do_cfg = do_cfg,
                )      
                if do_cfg:  
                    model_inputs = self.prepare_cfg_input(
                        model_inputs,
                        cfg_repeat_name_list = self.cfg_repeat_name_list,
                        neg_input_ids = model_kwargs.get(
                            'neg_input_ids', None
                        ) if (gen_loop_num == 0) else None,
                    )
                
                # output_token_num = torch.sum(self.unmatched_num[self.ongoing_row_list] + 1 \
                #                              + self.jacobi_init_num[self.ongoing_row_list],dim=0) if len(self.ongoing_row_list) > 0 else 1
                output_token_num =  model_input_ids.shape[1] if len(self.ongoing_row_list) > 0 else 1

                outputs = self(**model_inputs, return_dict=True)

                if synced_gpus and this_peer_finished:
                    continue  

                logits = outputs.logits # all tokens logits

                if model_kwargs['last_token_in_row_idx'] is not None:
                    logits = logits[:, 1:]
                    model_kwargs['last_token_in_row_idx'] = None
              
                next_tokens, next_token_scores = sampling_logits2tokens(
                    logits,
                    all_collected_input_ids,
                    unfinished_sequences, pad_token_id,
                    output_token_num = output_token_num, 
                    logits_processor=logits_processor, logits_warper=logits_warper,
                    do_sample=do_sample,
                    has_eos_stopping_criteria=has_eos_stopping_criteria,
                    do_cfg=do_cfg,
                    generator=self.generator, 
                    guidance_scale=guidance_scale,
                    is_force_no_cfg=is_force_no_cfg,
                )

                if do_cfg:
                    model_inputs = postprocess_cfg_decode(model_inputs) 
            
                (num_matched_tokens, num_unmatched_tokens, matched_next_tokens, unmatched_next_tokens, 
                matched_next_scores, unmatched_next_scores) = self.prefix_matching_next_tokens(
                    model_input_ids=model_input_ids,  # (batch_size, input_size)
                    input_token_scores = input_token_scores, # (batch_size, input_size, dim)
                    next_tokens=next_tokens,  # (batch_size, input_size)
                    next_token_scores=next_token_scores, #  (batch_size, input_size, dim)
                    prefix_token_sampler = prefix_token_sampler,
                    logits_processor = logits_processor, logits_warper = logits_warper,
                    all_collected_input_ids = all_collected_input_ids, # 已生成的序列
                )           

                additional_tokens = unmatched_next_tokens
                additional_scores = unmatched_next_scores
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs
                
                if len(self.ongoing_row_list) == 0:  
                    # update generated ids, model inputs, and length for next step
                    input_ids = torch.cat([input_ids, matched_next_tokens], dim=-1)
                    temporary_collected_scores = torch.cat([temporary_collected_scores, matched_next_scores], dim=-2)
                    
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                        num_new_tokens = output_token_num,
                    )
                
                else:
                    split_indices = num_matched_tokens[:-1].cumsum(0)

                    matched_token_parts = torch.tensor_split(matched_next_tokens, split_indices.tolist(), dim=1) 
                    matched_token_score_parts = torch.tensor_split(matched_next_scores, split_indices.tolist(), dim=1)

                    need_remove_row = []
                    delete_segments = []
            
                    for i, row in enumerate(self.ongoing_row_list):
                        matched_num = num_matched_tokens[i]
                        self.unmatched_num[row] = num_unmatched_tokens[i]
                        if self.unmatched_num[row] > 0: 
                            start_position_in_row = (
                                                        self.prompt_len + 2 
                                                        + torch.sum(self.keep_unmatched_num[:row], dim=-1) 
                                                        + torch.sum(self.jacobi_init_num[:row], dim=-1) 
                                                        + torch.sum(self.row_token_num[:row+1], dim=-1) - 1
                                                    )                      
                            start_unmatched_position_in_row = start_position_in_row + matched_num
                            end_unmatched_position_in_row = (
                                                    self.prompt_len + 2 
                                                    + torch.sum(self.keep_unmatched_num[:row+1], dim=-1) 
                                                    + torch.sum(self.jacobi_init_num[:row+1], dim=-1) 
                                                    + torch.sum(self.row_token_num[:row+1], dim=-1)
                                                  )
                            delete_segments.append([start_unmatched_position_in_row, end_unmatched_position_in_row])

                    if len(delete_segments) > 0:
                        delete_false_key_value(model_kwargs["past_key_values"], delete_segments, device)
                    
                    for index, (matched_token, matched_token_score, row) in enumerate(zip(matched_token_parts, matched_token_score_parts, self.ongoing_row_list)): 
                        matched_num = matched_token.shape[1]
                        position = torch.sum(self.row_token_num[:(row + 1)], dim=0) + self.prompt_len + 2 
                        input_ids = self.position_insert(input_ids, matched_token, position) 
                        temporary_collected_scores = self.position_insert(temporary_collected_scores, matched_token_score, position)

                        self.row_token_num[row] += matched_num 
                        
                        if index == 0:  
                            self.verify_num[row] = min(self.tokens_per_row-self.row_token_num[row], self.window_size)
                        else: 
                            self.verify_num[row] = min(self.row_token_num[row-1]-self.row_token_num[row], self.window_size)
                        
                        self.keep_unmatched_num[row] = self.unmatched_num[row]
                        if self.unmatched_num[row] >= self.verify_num[row] - 1:  
                            self.keep_unmatched_num[row] = self.verify_num[row] - 1 # >= -1
                        self.jacobi_init_num[row] = self.verify_num[row] - 1 - self.keep_unmatched_num[row] # >=0

                        if self.row_token_num[row] >= self.long_size and row < self.tokens_per_row - 2 \
                            and row+1 not in self.ongoing_row_list: ## append a eol token in this line to start next line  49-1-1 (0～47)
                    
                            input_ids = torch.cat([input_ids, self.eol_token[:, None]], dim=1)
                            eol_scores = torch.zeros((temporary_collected_scores.shape[0], 1, temporary_collected_scores.shape[-1]), 
                                                     dtype=temporary_collected_scores.dtype, device=temporary_collected_scores.device)
                            temporary_collected_scores = torch.cat([temporary_collected_scores, eol_scores], dim=1)

                            self.ongoing_row_list.append(row+1) 
                            self.row_token_num[row+1] = 1 
                            self.verify_num[row+1] = min(self.row_token_num[row]-self.row_token_num[row+1], self.window_size)
                            self.jacobi_init_num[row+1] = self.verify_num[row+1] - 1
                            self.keep_unmatched_num[row+1] = 0

                        elif row == self.tokens_per_row - 2 and self.row_token_num[row] == self.tokens_per_row: #全部结束 tokens_per_row = w+1
                            input_ids = torch.cat([input_ids, self.eol_token[:, None]], dim=1)
                            input_ids = torch.cat([input_ids, self.eoi_token[:, None]], dim=1)
                            input_ids = torch.cat([input_ids, self.end_token[:, None]], dim=1)
                            break

                        if self.row_token_num[row] == self.tokens_per_row: ## this row is done
                            model_kwargs['last_token_in_row_idx'] = position + matched_num - 1
                            self.verify_num[row] = 0
                            self.keep_unmatched_num[row] = 0
                            self.jacobi_init_num[row] = 0
                            need_remove_row.append(row)

                    for row in need_remove_row: 
                        self.ongoing_row_list.remove(row)
                    need_remove_row.clear()

                # check whether we get the end token
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

                cur_len = input_ids.shape[1]
                gen_loop_num += 1

                if (input_ids[0] == 8710).sum() >= 2:
                    this_peer_finished = True
                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            if streamer is not None:
                streamer.end()
            
            if count_time:
                t2.record()
                torch.cuda.synchronize()

                t = t1.elapsed_time(t2) / 1000
                print("Time elapsed inner: ", t)
                print("gen loop num (NFE): ", gen_loop_num)
                print("tokens length: ", cur_len)
                logging.info(f"Time elapsed inner: {t}")
                logging.info(f"gen loop num (NFE): {gen_loop_num}")
                logging.info(f"tokens length: {cur_len}")
            
            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids
    
    return JacobiSampler

def renew_backbone(model_class): #ChamaleonModel
    class JacobiBackbone(model_class):

        def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:  
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device) #用-inf填充
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)  #上三角
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1) #!!!大于当前位置的元素取True  0代表能够attend cache_position可能是 2 4 5 7
                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1) #所有batch共享一份mask  因果只负责位置上的mask，靠padding来屏蔽自定义的mask
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)

                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill( #仅关注位置上允许且不是padding的token
                        padding_mask, min_dtype
                    )
            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask
    
        def _update_causal_mask_JACOBI(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
            local_position_ids: torch.LongTensor,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1 #
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device) #用-inf填充
                indices = torch.arange(target_length, device=causal_mask.device).unsqueeze(0).expand(sequence_length, -1)
                if indices.shape[0] != local_position_ids.shape[1]:
                    pass
                before_mask = indices <= local_position_ids.squeeze_(0).unsqueeze_(1)
                causal_mask[before_mask] = 0
                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)

                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill( #仅关注位置上允许且不是padding的token
                        padding_mask, min_dtype
                    )
            
            return causal_mask
    
    return JacobiBackbone

def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)  #FlexAR
    pipe_line._init_new_params(**kwargs) 
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__) #Chameleon
    pipe_line.model._init_new_params(**kwargs)
    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__) #Backbone
    return pipe_line