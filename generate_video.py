import argparse
import gc
import os
import random
import time
import logging
import multiprocessing
import json
from typing import Dict, List

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

# IMPORTANT: Configure spawn method for multiprocessing
# This is critical for CUDA + multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def process_chunk(gpu_id: int, model_path: str, image_path: str, params: Dict, result_queue):
    """
    Procesează un chunk de cadre pe un GPU specific
    """
    try:
        # Setăm dispozitivul corect pentru acest proces
        torch.cuda.set_device(gpu_id)
        logger.info(f"[GPU {gpu_id}] Process started, initializing model...")
        
        # Încărcăm imaginea dacă este furnizată
        image = None
        if image_path:
            logger.info(f"[GPU {gpu_id}] Loading input image: {image_path}")
            image = load_image(image_path).convert("RGB")
            
            # Redimensionam imaginea la dimensiunile specificate
            if "width" in params and "height" in params:
                image = resizecrop(image, params["height"], params["width"])
                logger.info(f"[GPU {gpu_id}] Resized image to {params['width']}x{params['height']}")
        
        # Determinăm tipul de model (T2V sau I2V)
        is_t2v = "T2V" in model_path or image is None
        is_i2v = "I2V" in model_path or image is not None
        
        # Creăm pipeline-ul corespunzător
        logger.info(f"[GPU {gpu_id}] Initializing {'Text-to-Video' if is_t2v else 'Image-to-Video'} pipeline...")
        if is_t2v and not image:
            pipe = Text2VideoPipeline(
                model_path=model_path, 
                dit_path=model_path, 
                use_usp=False,  # Dezactivam USP pentru acest chunk
                offload=params.get("offload", False)
            )
        else:
            pipe = Image2VideoPipeline(
                model_path=model_path, 
                dit_path=model_path, 
                use_usp=False,  # Dezactivam USP pentru acest chunk
                offload=params.get("offload", False)
            )
        
        logger.info(f"[GPU {gpu_id}] Pipeline initialized successfully")
        
        # Configurăm TEACache dacă e necesar
        if params.get("teacache", False):
            logger.info(f"[GPU {gpu_id}] Initializing TEACache...")
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=params.get("num_inference_steps", 30),
                teacache_thresh=params.get("teacache_thresh", 0.2), 
                use_ret_steps=params.get("use_ret_steps", False),
                ckpt_dir=model_path
            )
        
        # Calculăm intervalul de cadre pentru acest GPU
        start_frame = params.get("gpu_start_frames", {}).get(str(gpu_id), 0)
        end_frame = params.get("gpu_end_frames", {}).get(str(gpu_id), params.get("num_frames", 97))
        frames_to_process = end_frame - start_frame
        
        logger.info(f"[GPU {gpu_id}] Processing frames {start_frame} to {end_frame-1} (total: {frames_to_process} frames)")
        
        # Adaptăm parametrii pentru acest chunk
        local_params = {
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", ""),
            "num_frames": frames_to_process,
            "num_inference_steps": params.get("num_inference_steps", 30),
            "guidance_scale": params.get("guidance_scale", 6.0),
            "shift": params.get("shift", 8.0),
            "generator": torch.Generator(device=f"cuda:{gpu_id}").manual_seed(params.get("seed", 0)),
            "height": params.get("height", 544),
            "width": params.get("width", 960),
        }
        
        # Adăugăm imaginea dacă este furnizată
        if image:
            local_params["image"] = image
        
        # Generăm cadrele
        logger.info(f"[GPU {gpu_id}] Starting generation...")
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(**local_params)[0]
        
        logger.info(f"[GPU {gpu_id}] Generated {len(video_frames)} frames successfully")
        
        # Punem rezultatele în coadă, inclusiv informațiile de ordonare
        result_queue.put({
            "gpu_id": gpu_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": video_frames
        })
        
        # Eliberăm memoria
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"[GPU {gpu_id}] Process completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Error in processing: {str(e)}")
        # Punem eroarea în coadă pentru a o gestiona în procesul principal
        result_queue.put({
            "gpu_id": gpu_id,
            "error": str(e)
        })
        return False

def generate_multi_gpu(model_path: str, image_path: str, params: Dict, gpu_ids: List[int]):
    """
    Generează un video folosind mai multe GPU-uri, fiecare procesând o parte din cadre
    """
    # Calculăm câte cadre va procesa fiecare GPU
    num_gpus = len(gpu_ids)
    total_frames = params.get("num_frames", 97)
    frames_per_gpu = total_frames // num_gpus
    remainder = total_frames % num_gpus
    
    # Distribuim cadrele către GPU-uri
    gpu_start_frames = {}
    gpu_end_frames = {}
    
    start_frame = 0
    for i, gpu_id in enumerate(gpu_ids):
        # Adăugăm un cadru în plus la primele 'remainder' GPU-uri
        extra_frame = 1 if i < remainder else 0
        frames_for_this_gpu = frames_per_gpu + extra_frame
        
        gpu_start_frames[str(gpu_id)] = start_frame
        gpu_end_frames[str(gpu_id)] = start_frame + frames_for_this_gpu
        
        start_frame += frames_for_this_gpu
    
    # Adăugăm informațiile de distribuție la parametri
    params_copy = params.copy()  # Creăm o copie pentru a nu modifica originalul
    params_copy["gpu_start_frames"] = gpu_start_frames
    params_copy["gpu_end_frames"] = gpu_end_frames
    
    logger.info(f"Frame distribution across GPUs: {json.dumps(gpu_start_frames)} to {json.dumps(gpu_end_frames)}")
    
    # Creăm o coadă pentru rezultate
    result_queue = multiprocessing.Queue()
    
    # Create processes for each GPU
    processes = []
    for gpu_id in gpu_ids:
        p = multiprocessing.Process(
            target=process_chunk,
            args=(gpu_id, model_path, image_path, params_copy, result_queue)
        )
        processes.append(p)
        p.start()
        logger.info(f"Started process for GPU {gpu_id}")
    
    # Așteptăm și colectăm rezultatele
    results = []
    for _ in range(len(gpu_ids)):
        result = result_queue.get()
        if "error" in result:
            logger.error(f"Error in GPU {result['gpu_id']}: {result['error']}")
        else:
            results.append(result)
    
    # Așteptăm ca toate procesele să se termine
    for p in processes:
        p.join()
    
    # Verificăm dacă avem toate rezultatele
    if len(results) < len(gpu_ids):
        raise RuntimeError(f"Only {len(results)} out of {len(gpu_ids)} GPUs completed successfully")
    
    # Sortăm rezultatele după start_frame pentru a reconstitui videoul complet
    results.sort(key=lambda x: x["start_frame"])
    
    # Combinăm cadrele din toate chunk-urile
    all_frames = []
    for result in results:
        all_frames.extend(result["frames"])
    
    logger.info(f"Successfully combined {len(all_frames)} frames from {len(results)} GPUs")
    
    return all_frames

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
    
    # Parsăm lista de GPU-uri pentru procesarea multi-GPU
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(",") if id.strip()]
    if not gpu_ids:
        gpu_ids = [0]
    
    # Verificăm dispozitivele CUDA disponibile
    num_available_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_available_gpus}")
    
    if num_available_gpus < len(gpu_ids):
        logger.warning(f"Requested {len(gpu_ids)} GPUs, but only {num_available_gpus} are available")
        gpu_ids = gpu_ids[:num_available_gpus]
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Inițializăm explicit fiecare GPU pentru a ne asigura că sunt disponibile
    for gpu_id in gpu_ids:
        try:
            with torch.cuda.device(gpu_id):
                torch.tensor([1.0], device=f"cuda:{gpu_id}")
                free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                logger.info(f"GPU {gpu_id}: {free_mem/1024**3:.2f}GB free out of {total_mem/1024**3:.2f}GB total")
        except Exception as e:
            logger.error(f"Error initializing GPU {gpu_id}: {str(e)}")
            raise RuntimeError(f"Failed to initialize GPU {gpu_id}")

    # Folosim negative_prompt din args
    negative_prompt = args.negative_prompt
    logger.info(f"Using negative prompt: {negative_prompt[:100]}...")

    prompt_input = args.prompt
    if args.prompt_enhancer:
        logger.info("Initializing and applying prompt enhancer...")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        logger.info(f"Enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    # Pregătim parametrii pentru generare
    generation_params = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "seed": args.seed,
        "height": height,
        "width": width,
        "offload": args.offload,
        "teacache": args.teacache,
        "teacache_thresh": args.teacache_thresh,
        "use_ret_steps": args.use_ret_steps
    }

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Starting video generation with {args.num_frames} frames...")
    generation_start = time.time()
    
    # Utilizăm multiple GPU-uri
    if len(gpu_ids) > 1:
        logger.info(f"Using custom multi-GPU approach with {len(gpu_ids)} GPUs")
        try:
            video_frames = generate_multi_gpu(args.model_id, args.image, generation_params, gpu_ids)
        except Exception as e:
            logger.error(f"Multi-GPU processing failed: {str(e)}")
            logger.warning("Falling back to single GPU mode")
            gpu_ids = [gpu_ids[0]]  # Folosim doar primul GPU
    
    # Dacă avem doar un GPU sau multi-GPU a eșuat
    if len(gpu_ids) == 1:
        # Utilizăm un singur GPU
        logger.info(f"Using single GPU mode on GPU {gpu_ids[0]}")
        # Setăm GPU-ul corect
        torch.cuda.set_device(gpu_ids[0])
        
        image = None
        if args.image:
            logger.info(f"Loading input image: {args.image}")
            image = load_image(args.image).convert("RGB")
            logger.info("Image loaded successfully")

        # Creăm pipeline-ul pentru acest GPU
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
            
            # Prepare image if needed
            image_width, image_height = image.size
            if args.width is None and args.height is None and image_height > image_width:
                logger.info(f"Portrait image detected, swapping dimensions")
                height, width = width, height
                
            logger.info(f"Resizing image to {width}x{height}")
            image = resizecrop(image, height, width)
        
        if args.teacache:
            logger.info("Initializing TEACache for accelerated inference...")
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=args.inference_steps, 
                teacache_thresh=args.teacache_thresh, 
                use_ret_steps=args.use_ret_steps, 
                ckpt_dir=args.model_id
            )
            logger.info(f"TEACache initialized with threshold={args.teacache_thresh}, use_ret_steps={args.use_ret_steps}")
        
        # Pregătim parametrii pentru generare
        single_gpu_params = {
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
            single_gpu_params["image"] = image
        
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            logger.info(f"Inference parameters: guidance_scale={args.guidance_scale}, shift={args.shift}, steps={args.inference_steps}")
            logger.info(f"Generation in progress... This process may take several minutes.")
            
            try:
                video_frames = pipe(**single_gpu_params)[0]
            except Exception as e:
                logger.error(f"Error in single GPU mode: {str(e)}")
                raise
    
    generation_time = time.time() - generation_start
    logger.info(f"Video generation completed in {generation_time:.2f} seconds")

    logger.info("Saving result video...")
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
    output_path = os.path.join(save_dir, video_out_file)
    imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
    logger.info(f"Video saved at: {output_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Complete process took {total_time:.2f} seconds")
    logger.info(f"Generation statistics: {args.num_frames} frames, {generation_time:.2f} seconds, {args.num_frames/generation_time:.2f} frames/second")
