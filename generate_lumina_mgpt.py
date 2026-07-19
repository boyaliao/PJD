import os

from argparse import ArgumentParser
from datetime import datetime

from dataset_tools.dataset_templates import create_dataset
from lumina_mgpt.inference_solver import FlexARInferenceSolver
from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
from utils import set_logger


def get_exp_name(
    model,
    target_size,
    window_size,
    long_size,
    gpu_id,
    note=None,

):
    date_str = datetime.now().strftime("%Y%m%d")
    name_parts = [
        f"model={model}",
        f"target_size={target_size}",
        f"window_size={window_size}",
        f"long_size={long_size}",
        f"gpu_id={gpu_id}",
        f"date={date_str}",
    ]
    if note:
        name_parts.append(f"note={note}")

    return "_".join(name_parts)


if __name__ == "__main__":
    # set start method as 'spawn' to avoid CUDA re-initialization issues
    # multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument("--savedir", type=str, default="./workdir") 
    parser.add_argument("--expdir",  type=str, default="./expdir") 
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="parti_cocoformat"
    )
    parser.add_argument(
        "--dataset_anno_file", 
        type=str, 
        default="/parti-prompts/PartiPrompts_sample_200.tsv"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--gpu_ids", 
        type=int, 
        nargs="+", 
        default=[0], 
        help="List of GPU ids, e.g. --gpu_ids 0 1 2 3"
    )
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument(
        "--node_ids", 
        type=int, 
        nargs="+", 
        default=[0], 
        help="List of node ids, e.g. --node_ids 0 1 2 3"
    )

    parser.add_argument("--cache_dir", type=str, default="./ckpts/")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/liaoboya/Datasets/Lumina-mGPT-7B-768",
    )
    parser.add_argument("--model_name", type=str, default="Lumina-mGPT-7B")
    parser.add_argument("--target_size", type=int, default=768)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--long_size", type=int, default=11)
    parser.add_argument("--note", type=str)

    args = parser.parse_args()

    savedir = args.savedir
    expdir = args.expdir
    ann_file = args.dataset_anno_file
    dataset_name = args.dataset_name

    gpu_id = args.gpu_id
    gpu_ids = args.gpu_ids
    node_id = args.node_id
    node_ids = args.node_ids

    dataset_params = dict(
		name = dataset_name,
		annFile = ann_file,
        ds_type = 'eval',
	)
   
    cache_dir = args.cache_dir
    model_path = args.model_path
    target_size = args.target_size

    target_size_h = target_size
    target_size_w = target_size

    device = f"cuda:0"   
    print("device:", device)

    sub_dir = get_exp_name(
                model="Lumina-mGPT-7B",
                target_size=768,
                window_size=16,
                long_size=11,
                gpu_id=gpu_id,
                note = "PJD_local"
                )
    log_name = sub_dir + '.log'
    savedir = os.path.join(savedir, sub_dir)
    log_file = os.path.join(expdir, log_name)
    print(f"save_dir:{savedir}")
    print(f"exp_file:{expdir}")

    os.makedirs(savedir, exist_ok=True)
    set_logger(log_level='info', fname=log_file) 

    ds = create_dataset(  
        gpu_id=gpu_id,
        gpu_ids=gpu_ids,
        node_id=node_id,
        node_ids=node_ids,
		**dataset_params,
	)

    # ******************** Image Generation ********************
    inference_solver = FlexARInferenceSolver(  
        model_path=model_path,
        precision="bf16",
        tokenizer=model_path,
        target_size=target_size,
        cache_dir=cache_dir,
        device = device,
    )
    seeds = [None, ] 
    max_num_new_tokens = 16
    multi_token_init_scheme = 'random' # 'repeat_horizon'
    image_top_k = 2000
    text_top_k = 10
    guidance_scale = 3.0
    prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

    from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
    inference_solver = renew_pipeline_sampler(   
        inference_solver,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        target_size=target_size
    )

    template_condition_sentence = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"

    for i in range(len(ds)):
        prompt = ds[i]
        q1 = template_condition_sentence + prompt['caption']
        file_name = str(prompt['image_id']).zfill(12) + ".jpg"
        output_file_path = os.path.join(savedir, file_name)

        generated = inference_solver.generate(
            images=[],
            qas=[[q1, None]], 
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
        )

        a1, new_image = generated[0], generated[1][0]

        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        result_image.save(output_file_path)
    
    del inference_solver