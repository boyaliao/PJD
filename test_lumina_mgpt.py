import gc
import random
import time

import numpy as np
import torch

from pathlib import Path
from lumina_mgpt.inference_solver import FlexARInferenceSolver

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

MODEL_CONFIGS = {
    "7b-512": {
        "model_path": "Alpha-VLLM/Lumina-mGPT-7B-512",
        "target_size": 512,
    },
    "7b-768": {
        "model_path": "Alpha-VLLM/Lumina-mGPT-7B-768",
        "target_size": 768,
    },
    "7b-1024": {
        "model_path": "Alpha-VLLM/Lumina-mGPT-7B-1024",
        "target_size": 1024,
    },
}

model_name = "7b-768"
config = MODEL_CONFIGS[model_name]
cache_dir = Path("ckpts")
save_dir = Path("ours_result")
save_dir.mkdir(parents=True, exist_ok=True)
model_path = config["model_path"]
target_size = config["target_size"]
target_size_h = target_size
target_size_w = target_size

device = torch.device("cuda:0")

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path=model_path,
    precision="bf16",
    target_size=target_size,
    cache_dir=cache_dir,
    device = device,
)

seeds = [None, ] #[None, ] 
max_num_new_tokens = 8
multi_token_init_scheme = 'random' # 'repeat_horizon'
image_top_k = 2000 
text_top_k = 10
guidance_scale = 3.0
prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

q_image_content_conditions = [
    "Vibrant oil painting of a cozy countryside kitchen interior, warm sunlight, rich textures, thick impasto strokes, high saturation",
    "Bold oil painting of a woman in red dress, swirling brush strokes, dramatic lighting, expressive saturated colors on textured canvas",
    "Color-rich oil painting of a peaceful river reflecting a glowing sunset, heavy strokes, vivid oranges and purples, impressionist style",
]

template_condition_sentences = [
    f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n",
] * len(q_image_content_conditions)

from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
print(inference_solver.__class__)
inference_solver = renew_pipeline_sampler(
    inference_solver,
    max_num_new_tokens=max_num_new_tokens,
    guidance_scale=guidance_scale,
    seed=seeds[0],
    multi_token_init_scheme=multi_token_init_scheme,
    do_cfg=True,
    image_top_k=image_top_k, 
    text_top_k=text_top_k,
    prefix_token_sampler_scheme=prefix_token_sampler_scheme,
    target_size=target_size,
)

for seed in seeds:
    inference_solver.model.seed = seed
    for i, q_image_content_condition in enumerate(q_image_content_conditions):
        q1 = template_condition_sentences[i] + q_image_content_condition

        output_file_name = (
            f"{model_name}-{q_image_content_condition[:30]}"
            f"-{max_num_new_tokens}"
            f"-init-{multi_token_init_scheme[:6]}"
            f"-seed-{seed}"
            f"-img-topk-{image_top_k}.png"
        )

        time_start = time.time()
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t1.record()

        generated = inference_solver.generate(
            images=[],
            qas=[[q1, None]], 
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
        )
        t2.record()
        torch.cuda.synchronize()

        t = t1.elapsed_time(t2) / 1000
        time_end = time.time()
        print("Time elapsed: ", t, time_end - time_start)

        a1, new_image = generated[0], generated[1][0]

        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        output_path = save_dir / output_file_name
        result_image.save(output_path)
        print(f"{a1} saved to {output_path}") # <|image|>

del inference_solver
gc.collect()