import argparse
import gc
import os
import random
import time
import logging

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


if __name__ == "__main__":
    logger.info("SkyReels-V2 Video Generator - Starting")
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-T2V-14B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
    parser.add_argument("--width", type=int, default=None, help="Custom width (overrides default for resolution)")
    parser.add_argument("--height", type=int, default=None, help="Custom height (overrides default for resolution)")
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    # Adăugat parametrul pentru negative_prompt
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt to specify what you don't want in the video"
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    # Adăugăm parametrii pentru multi-GPU
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use for multi-GPU processing (e.g. '0,1')")
    args = parser.parse_args()
    
    logger.info(f"Configuration: Resolution={args.resolution}, Frames={args.num_frames}, Steps={args.inference_steps}")
    if args.image:
        logger.info(f"Mode: Image-to-Video with file {args.image}")
    else:
        logger.info(f"Mode: Text-to-Video")
    
    logger.info(f"Downloading/verifying model: {args.model_id}")
    args.model_id = download_model(args.model_id)
    logger.info(f"Model loaded: {args.model_id}")

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))
    logger.info(f"Seed used: {args.seed}")

    # Set dimensions based on resolution and custom width/height if provided
    if args.resolution == "540P":
        height = args.height or 544
        width = args.width or 960
    elif args.resolution == "720P":
        height = args.height or 720
        width = args.width or 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")
    
    logger.info(f"Video dimensions: {width}x{height}")

    image = None
    if args.image:
        logger.info(f"Loading input image: {args.image}")
        image = load_image(args.image).convert("RGB")
        logger.info("Image loaded successfully")

    # Folosim negative_prompt din args în loc de hardcodat
    negative_prompt = args.negative_prompt
    
    local_rank = 0
    if args.use_usp:
        logger.info("Initializing USP mode for multi-GPU processing")
        assert not args.prompt_enhancer, "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        # Procesăm lista de GPU IDs
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(",") if id.strip()]
        if not gpu_ids:
            gpu_ids = [0]  # Implicit folosim GPU 0
        
        # Setăm variabilele de mediu pentru a restricționa dispozitivele CUDA vizibile
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={args.gpu_ids}")
        
        # Afișăm informații despre fiecare GPU specificat
        total_memory = 0
        free_memory = 0
        logger.info(f"Checking memory for GPUs: {args.gpu_ids}")
        
        # Verificăm numărul de GPU-uri vizibile acum
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of visible GPUs: {num_gpus}")
        
        for device_idx in range(num_gpus):
            try:
                # Selectăm explicit dispozitivul curent
                torch.cuda.set_device(device_idx)
                free_mem, total_mem = torch.cuda.mem_get_info()
                logger.info(f"GPU {device_idx}: {free_mem/1024**3:.2f}GB free out of {total_mem/1024**3:.2f}GB total")
                
                # Calculăm totalul
                total_memory += total_mem
                free_memory += free_mem
            except Exception as e:
                logger.error(f"Error checking GPU {device_idx}: {e}")
        
        logger.info(f"Combined GPU memory: {free_memory/1024**3:.2f}GB free out of {total_memory/1024**3:.2f}GB total")

        # Inițializăm procesul de grup pentru procesare distribuită
        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        logger.info(f"USP initialized with rank {local_rank}, world_size {dist.get_world_size()}, using GPUs: {args.gpu_ids}")

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        logger.info("Initializing and applying prompt enhancer...")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        logger.info(f"Enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    # GPU memory before loading the model
    if torch.cuda.device_count() > 0:
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"VRAM before loading model (primary GPU): {free_mem/1024**3:.2f}GB free out of {total_mem/1024**3:.2f}GB total")

    if image is None:
        assert "T2V" in args.model_id, f"check model_id:{args.model_id}"
        logger.info("Initializing Text-to-Video pipeline...")
        pipe = Text2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
        logger.info("Text-to-Video pipeline initialized successfully")
    else:
        assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
        logger.info("Initializing Image-to-Video pipeline...")
        pipe = Image2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
        logger.info("Image-to-Video pipeline initialized successfully")
        args.image = load_image(args.image)
        image_width, image_height = args.image.size
        
        # Only do auto-swap if custom dimensions weren't provided
        if args.width is None and args.height is None and image_height > image_width:
            logger.info(f"Portrait image detected, swapping dimensions")
            height, width = width, height
            
        logger.info(f"Resizing image to {width}x{height}")
        args.image = resizecrop(args.image, height, width)

    # GPU memory after loading the model
    if torch.cuda.device_count() > 0:
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"VRAM after loading model (primary GPU): {free_mem/1024**3:.2f}GB free out of {total_mem/1024**3:.2f}GB total")

    if args.teacache:
        logger.info("Initializing TEACache for accelerated inference...")
        pipe.transformer.initialize_teacache(enable_teacache=True, num_steps=args.inference_steps, 
                                             teacache_thresh=args.teacache_thresh, use_ret_steps=args.use_ret_steps, 
                                             ckpt_dir=args.model_id)
        logger.info(f"TEACache initialized with threshold={args.teacache_thresh}, use_ret_steps={args.use_ret_steps}")
        
    # Afișăm și negative prompt în log
    logger.info(f"Using negative prompt: {negative_prompt[:100]}...")
    
    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "height": height,
        "width": width,
    }

    if image is not None:
        kwargs["image"] = args.image.convert("RGB")

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Starting video generation with {args.num_frames} frames...")
    generation_start = time.time()
    
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        logger.info(f"Inference parameters: guidance_scale={args.guidance_scale}, shift={args.shift}, steps={args.inference_steps}")
        
        # Manual logging instead of callback since the callback isn't supported
        logger.info(f"Generation in progress... This process may take several minutes.")
        
        video_frames = pipe(**kwargs)[0]
    
    generation_time = time.time() - generation_start
    logger.info(f"Video generation completed in {generation_time:.2f} seconds")

    if local_rank == 0:
        logger.info("Saving result video...")
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
        logger.info(f"Video saved at: {output_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Complete process took {total_time:.2f} seconds")
    logger.info(f"Generation statistics: {args.num_frames} frames, {generation_time:.2f} seconds, {args.num_frames/generation_time:.2f} frames/second")
