#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from glob import glob
import threading
import gradio as gr
from PIL import Image
import torch
import numpy as np
import signal
from contextlib import contextmanager
from transformers import pipeline

# -- CONFIG -------------------------------------------------------------------

MODEL_DIR = "./models"
VIDEO_OUTPUT_DIR = "./result/video_out"
IMAGE_OUTPUT_DIR = "./image_out"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Negative prompt predefinit (poate fi modificat de utilizator)
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

ENGLISH_NEGATIVE_PROMPT = "Vibrant colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall greyness, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless scene, cluttered background, three legs, crowded background, walking backwards"

# Define all model categories
MODEL_INFO = {
    "SkyReels-V2-T2V-14B-540P": {
        "type": "T2V", 
        "repo_id": "Skywork/SkyReels-V2-T2V-14B-540P",
        "desc": "Text-to-Video model (14B parameters, 540P resolution)"
    },
    "SkyReels-V2-T2V-14B-720P": {
        "type": "T2V", 
        "repo_id": "Skywork/SkyReels-V2-T2V-14B-720P",
        "desc": "Text-to-Video model (14B parameters, 720P resolution)"
    },
    "SkyReels-V2-I2V-1.3B-540P": {
        "type": "I2V", 
        "repo_id": "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "desc": "Image-to-Video model (1.3B parameters, 540P resolution)"
    },
    "SkyReels-V2-I2V-14B-540P": {
        "type": "I2V", 
        "repo_id": "Skywork/SkyReels-V2-I2V-14B-540P",
        "desc": "Image-to-Video model (14B parameters, 540P resolution)"
    },
    "SkyReels-V2-I2V-14B-720P": {
        "type": "I2V", 
        "repo_id": "Skywork/SkyReels-V2-I2V-14B-720P",
        "desc": "Image-to-Video model (14B parameters, 720P resolution)"
    },
    "SkyReels-V2-DF-1.3B-540P": {
        "type": "DF", 
        "repo_id": "Skywork/SkyReels-V2-DF-1.3B-540P",
        "desc": "Diffusion Forcing model for long videos (1.3B parameters, 540P resolution)"
    },
    "SkyReels-V2-DF-14B-540P": {
        "type": "DF", 
        "repo_id": "Skywork/SkyReels-V2-DF-14B-540P",
        "desc": "Diffusion Forcing model for long videos (14B parameters, 540P resolution)"
    },
    "SkyReels-V2-DF-14B-720P": {
        "type": "DF", 
        "repo_id": "Skywork/SkyReels-V2-DF-14B-720P",
        "desc": "Diffusion Forcing model for long videos (14B parameters, 720P resolution)"
    }
}

# Map of aspect ratios to dimensions
ASPECT_RATIOS = {
    "16:9 (Landscape)": {"540P": (960, 544), "720P": (1280, 720)},
    "9:16 (Portrait)": {"540P": (544, 960), "720P": (720, 1280)},
    "4:5 (Portrait)": {"540P": (544, 680), "720P": (720, 900)},
    "1:1 (Square)": {"540P": (544, 544), "720P": (720, 720)},
}

# -- HELPER FUNCTIONS --------------------------------------------------------

def get_available_models():
    """Discover all local models under ./models"""
    # Verifică și directoarele de tipul I2V
    all_dirs = glob(os.path.join(MODEL_DIR, "SkyReels-V2-*")) + \
               glob(os.path.join(MODEL_DIR, "*/SkyReels-V2-*"))
    
    model_map = {}
    for p in all_dirs:
        base_name = os.path.basename(p)
        if base_name not in model_map:  # evită duplicatele
            model_map[base_name] = p
    
    # Split into model types - allow DF models to be used for both T2V and I2V
    t2v_models = [m for m in model_map if "-T2V-" in m]
    i2v_models = [m for m in model_map if "-I2V-" in m]
    df_models = [m for m in model_map if "-DF-" in m]
    
    # Add DF models to both T2V and I2V lists
    t2v_models.extend(df_models)
    i2v_models.extend(df_models)
    
    # Afișează pentru debugging
    print(f"Found models: {list(model_map.keys())}")
    print(f"T2V models: {t2v_models}")
    print(f"I2V models: {i2v_models}")
    print(f"DF models: {df_models}")
    
    return model_map, t2v_models, i2v_models

def get_missing_models():
    """Find which models are not downloaded yet"""
    model_map, _, _ = get_available_models()
    return [name for name in MODEL_INFO.keys() if name not in model_map]

def download_model_huggingface(model_name, progress_callback=None):
    """Download a model from Hugging Face with progress updates"""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    
    repo_id = MODEL_INFO[model_name]["repo_id"]
    local_dir = os.path.join(MODEL_DIR, model_name)
    
    print(f"Downloading {model_name}...")
    os.makedirs(local_dir, exist_ok=True)
    
    class ProgressCallback:
        def __init__(self):
            self.pct = 0
            self.downloaded_bytes = 0
            self.total_bytes = 0
            self.status = f"Preparing to download {model_name}..."

        def __call__(self, progress):
            if progress.total:
                self.pct = round(progress.completed * 100 / progress.total, 1)
                self.downloaded_bytes = progress.completed
                self.total_bytes = progress.total
                self.status = f"Downloading {model_name}: {self.pct}% ({self.format_size(self.downloaded_bytes)}/{self.format_size(self.total_bytes)})"
                if progress_callback:
                    progress_callback(self.status)
                print(self.status)

        def format_size(self, size_bytes):
            """Format bytes to human readable size"""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes/(1024*1024):.1f} MB"
            else:
                return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    callback = ProgressCallback()
    
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir, 
            resume_download=True,
            max_workers=8,
            tqdm_class=None,
        )
        print(f"Successfully downloaded {model_name}")
        return f"✅ Successfully downloaded {model_name}"
    except HfHubHTTPError as e:
        error_msg = f"❌ Failed to download {model_name}: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Failed to download {model_name}: {str(e)}"
        print(error_msg)
        return error_msg

def download_selected_models(selected_models, progress_callback=None):
    """Download multiple selected models"""
    results = []
    total = len(selected_models)
    
    for i, model in enumerate(selected_models):
        status = f"Starting download {model} ({i+1}/{total})..."
        if progress_callback:
            progress_callback(status)
        print(status)
        
        result = download_model_huggingface(model, progress_callback)
        results.append(result)
        
        status = f"Completed {model} ({i+1}/{total})"
        if progress_callback:
            progress_callback(status)
        print(status)
    
    # Refresh the model lists after downloading
    model_map, t2v_models, i2v_models = get_available_models()
    
    return "\n".join(results)

def find_latest_video():
    """Find the latest generated video file."""
    mp4s = glob(os.path.join(VIDEO_OUTPUT_DIR, "*.mp4"))
    return max(mp4s, key=os.path.getctime) if mp4s else None

def parse_resolution_and_aspect(resolution, aspect_ratio):
    """Parse the resolution and aspect ratio to return width and height"""
    try:
        width, height = ASPECT_RATIOS[aspect_ratio][resolution]
        return width, height
    except:
        # Default to landscape if something went wrong
        if resolution == "540P":
            return 960, 544
        else:
            return 1280, 720

# which kwargs are boolean flags (no value)
FLAG_ONLY_ARGS = {
    "offload",
    "teacache",
    "use_ret_steps",
    "use_usp",
    "causal_attention",
    "prompt_enhancer",
    "cfg_rescale"
}

def build_command(script, model_path, resolution, aspect_ratio, num_frames, fps, prompt, negative_prompt=None, **kwargs):
    """Build a subprocess command list with resolution and aspect ratio support."""
    # Get width and height based on aspect ratio
    width, height = parse_resolution_and_aspect(resolution, aspect_ratio)
    
    cmd = [
        sys.executable, script,
        "--model_id", model_path,
        "--resolution", resolution,
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--prompt", prompt,
        "--outdir", "video_out"
    ]

        # Add multi-GPU specific params
    if "use_multi_gpu" in kwargs and kwargs["use_multi_gpu"]:
        gpu_ids = kwargs.get("gpu_devices", "0,1")
        cmd.extend(["--gpu_ids", gpu_ids])
        # Always enable USP for multi-GPU
        if "--use_usp" not in cmd:
            cmd.append("--use_usp")
            
    # Add negative prompt if provided
    if negative_prompt:
        cmd.extend(["--negative_prompt", negative_prompt])
    
    # Add the width and height parameters directly
    cmd.extend(["--width", str(width), "--height", str(height)])
    
    # Mapează steps la inference_steps dacă există
    if "steps" in kwargs:
        kwargs["inference_steps"] = kwargs.pop("steps")
    
    for key, value in kwargs.items():
        if key in FLAG_ONLY_ARGS:
            if value:
                cmd.append(f"--{key}")
        else:
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
    return cmd

def run_and_yield_logs(cmd):
    """
    Run a command and yield its output line by line in real-time.
    This allows us to display live logs in the Gradio interface.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip()
    proc.wait()

# -- PROMPT ENHANCER FUNCTION -------------------------------------------------

def run_prompt_enhancer(prompt):
    """Run the prompt enhancer script to enhance a basic prompt."""
    try:
        cmd = [
            sys.executable,
            "skyreels_v2_infer/pipelines/prompt_enhancer.py",
            "--prompt", prompt
        ]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Extract the enhanced prompt from the output
        output_lines = result.stdout.split('\n')
        enhanced_prompt = ""
        for line in output_lines:
            if line.startswith('Enhanced prompt:'):
                enhanced_prompt = line.replace('Enhanced prompt:', '').strip()
                break
        
        return enhanced_prompt if enhanced_prompt else result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running prompt enhancer: {e.stderr}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# -- MISTRAL PROMPT ENHANCER FUNCTION (IMPROVED v2) -------------------------------------------

_mistral_pipe = None

class TimeoutError(Exception):
    """Excepție personalizată pentru timeout"""
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=60):
    """Rulează o funcție cu timeout folosind threading în loc de signal"""
    result = [None]
    error = [None]
    completed = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True  
    
    thread.start()
    thread.join(timeout_seconds)
    
    if not completed[0]:
        if thread.is_alive():
            print(f"Operation timed out after {timeout_seconds} seconds!")
            return None, TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if error[0] is not None:
        return None, error[0]
        
    return result[0], None

def get_mistral_enhancer(nsfw_allowed=False, timeout_seconds=300):
    """Lazy-load the Mistral prompt enhancer model with explicit NSFW handling.
    
    Args:
        nsfw_allowed (bool): If True, will allow processing of NSFW prompts with explicit warnings.
                             If False, will reject NSFW prompts.
        timeout_seconds (int): Timpul maxim în secunde pentru generare
    """
    global _mistral_pipe
    
    try:
        # Check if model exists locally
        import os
        model_path = os.path.join("models", "mistral-7b-instruct-v0.2")
        if os.path.exists(model_path):
            print(f"Loading Mistral model from local path: {model_path}")
            model_source = model_path
        else:
            print("Downloading Mistral model from Hugging Face...")
            model_source = "mistralai/Mistral-7B-Instruct-v0.2"
        
        def generate_with_nsfw_check(prompt, **kwargs):
            global _mistral_pipe
            # NSFW content detection (simple example - would need more robust checking)
            nsfw_keywords = ['nsfw', 'explicit', 'adult', 'porn', 'sexual', 'nude']
            is_nsfw = any(keyword in prompt.lower() for keyword in nsfw_keywords)
            
            if is_nsfw and not nsfw_allowed:
                return "[BLOCKED] This prompt contains NSFW content which is not allowed in the current configuration."
            
            # Add explicit NSFW instruction if allowed
            if is_nsfw and nsfw_allowed:
                prompt = f"[EXPLICIT CONTENT ENABLED] {prompt}"
                print("WARNING: Generating content with explicit NSFW prompt")
            
            # Create the pipeline if it doesn't exist
            if _mistral_pipe is None:
                print("Creating Mistral pipeline...")
                try:
                    # Verificare memorie disponibilă (simplificată)
                    if torch.cuda.is_available():
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        print(f"Free GPU memory: {free_memory / (1024**3):.2f} GB")
                        if free_memory < 8 * (1024**3):  # Sub 8GB
                            print("WARNING: Low GPU memory. Using 8-bit quantization.")
                            kwargs["load_in_8bit"] = True
                    
                    # Folosim funcția noastră cu timeout
                    print("Starting pipeline creation (timeout: 120s)...")
                    create_pipeline = lambda: pipeline(
                        "text-generation",
                        model=model_source,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        max_new_tokens=256,
                    )
                    
                    pipe, error = run_with_timeout(create_pipeline, timeout_seconds=120)
                    
                    if error:
                        print(f"Error creating pipeline: {str(error)}")
                        return f"[ERROR] Could not create pipeline: {str(error)}"
                    
                    _mistral_pipe = pipe
                    print("Pipeline created successfully!")
                    
                except Exception as e:
                    print(f"Error creating pipeline: {str(e)}")
                    return f"[ERROR] Could not create pipeline: {str(e)}"
            
            print(f"Generating text with prompt length: {len(prompt)} characters...")
            start_time = time.time()
            
            try:
                # Funcția pentru generarea textului
                def generate_text():
                    return _mistral_pipe(prompt, **kwargs)
                
                # Rulăm cu timeout
                result, error = run_with_timeout(generate_text, timeout_seconds=timeout_seconds)
                
                if error:
                    if isinstance(error, TimeoutError):
                        print(f"Generation timed out after {timeout_seconds} seconds!")
                        return "[TIMEOUT] Text generation took too long and was interrupted."
                    else:
                        print(f"Error during generation: {str(error)}")
                        return f"[ERROR] {str(error)}"
                
                print(f"Generation completed in {time.time() - start_time:.2f} seconds")
                return result
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                return f"[ERROR] {str(e)}"
            
        return generate_with_nsfw_check
    except Exception as e:
        print(f"Error loading Mistral model: {str(e)}")
        return None

def enhance_prompt_with_mistral(text):
    """Generator function for enhancing prompts with Mistral."""
    if not text.strip():
        yield text, "⚠️ Please enter a prompt first!"
        return
    
    status_message = "🔄 Loading Mistral model and enhancing prompt..."
    yield text, status_message  # Returnează mesajul inițial de status
    
    try:
        # Lazy-load modelul la prima utilizare
        enhancer = get_mistral_enhancer(timeout_seconds=60)  # Timeout de 60 secunde pentru generare
        if enhancer is None:
            yield text, "❌ Error: Could not load Mistral model. Check console for details."
            return
        
        system_msg = "Rewrite the prompt for high‑quality AI video; keep meaning, add cinematic detail, concise English."
        prompt = f"<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{text} [/INST]"
        
        # Adaugă feedback intermediar pentru utilizator
        yield text, "🔄 Model loaded. Generating enhanced prompt..."
        
        resp = enhancer(prompt, temperature=0.7, do_sample=True, max_new_tokens=256)
        
        # Verifică dacă resp este string (în caz de eroare) sau lista de rezultate
        if isinstance(resp, str):
            # Deja avem un mesaj de eroare
            yield text, f"⚠️ {resp}"
            return
            
        # Verifică dacă avem un rezultat valid
        if not resp or not isinstance(resp, list) or len(resp) == 0 or "generated_text" not in resp[0]:
            yield text, "⚠️ Enhancement failed! Original prompt retained."
            return
            
        generated_text = resp[0]["generated_text"]
        
        # Verifică dacă avem [/INST] în text
        if "[/INST]" not in generated_text:
            yield text, "⚠️ Invalid response format. Original prompt retained."
            return
            
        enhanced = generated_text.split("[/INST]")[-1].strip()
        
        # Verifică dacă rezultatul nu este gol
        if not enhanced:
            yield text, "⚠️ Enhancement failed! Original prompt retained."
            return
        
        yield enhanced, "✅ Prompt enhanced successfully with Mistral-7B!"
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")
        yield text, f"❌ Error enhancing prompt: {str(e)}"

# Exemplu de utilizare (comentat, de decomenteat pentru testare):
"""
if __name__ == "__main__":
    test_prompt = "A futuristic city with flying cars"
    generator = enhance_prompt_with_mistral(test_prompt)
    
    # Afișează fiecare stare intermediară
    for prompt, status in generator:
        print(f"Status: {status}")
        print(f"Current prompt: {prompt}")
        print("-" * 50)
"""

# -- GENERATION FUNCTIONS -----------------------------------------------------

def text_to_video_live(
    prompt, negative_prompt, model_name, resolution, aspect_ratio, num_frames, fps,
    guidance_scale, shift, steps, seed, offload, use_usp, cfg_rescale,
    prompt_enhancer, causal_attention,
    ar_step=None, base_num_frames=None, overlap_history=None, addnoise_condition=None
):
    """Generate a video from text input with live logs."""
    if not model_name:
        return None, "Please select a model first!"
    
    model_path = os.path.join(MODEL_DIR, model_name)
    is_df = model_name.startswith("SkyReels-V2-DF-")
    script = "generate_video_df.py" if is_df else "generate_video.py"

    # Clamp DF parameters so they never exceed num_frames
    if is_df:
        base_num_frames = min(base_num_frames or num_frames, num_frames)
        overlap_history = min(overlap_history or 0, max(0, num_frames - 1))

    cmd = build_command(
        script, model_path, resolution, aspect_ratio, num_frames, fps, prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        # core args
        guidance_scale=(None if is_df else guidance_scale),
        shift=(None if is_df else shift),
        steps=steps,  # Acesta va fi mapat la inference_steps în build_command
        seed=seed,
        # DF-only args
        ar_step=(ar_step if is_df else None),
        base_num_frames=(base_num_frames if is_df else None),
        overlap_history=(overlap_history if is_df else None),
        addnoise_condition=(addnoise_condition if is_df else None),
        # flags
        offload=offload,
        use_usp=use_usp,
        cfg_rescale=cfg_rescale,
        prompt_enhancer=prompt_enhancer,
        causal_attention=causal_attention
    )

    logs = ""
    for line in run_and_yield_logs(cmd):
        logs += line + "\n"
        yield None, logs

    out = find_latest_video()
    yield out, logs

def image_to_video_live(
    image, prompt, negative_prompt, model_name, resolution, aspect_ratio, num_frames, fps,
    guidance_scale, shift, steps, seed, offload, teacache, use_ret_steps, teacache_thresh, 
    use_usp, cfg_rescale, prompt_enhancer, causal_attention,
    # DF model parameters
    ar_step=None, base_num_frames=None, overlap_history=None, addnoise_condition=None
):
    """Generate a video from an image input with live logs."""
    if not model_name:
        return None, "Please select a model first!"
    
    if image is None:
        return None, "Please upload an image first!"
    
    img_path = os.path.join(IMAGE_OUTPUT_DIR, "temp_input.jpg")
    if isinstance(image, Image.Image):
        image.save(img_path)
    else:
        Image.fromarray(image).save(img_path)

    # Check if it's a DF model
    is_df = model_name.startswith("SkyReels-V2-DF-")
    script = "generate_video_df.py" if is_df else "generate_video.py"
    
    # Clamp DF parameters so they never exceed num_frames
    if is_df:
        base_num_frames = min(base_num_frames or num_frames, num_frames)
        overlap_history = min(overlap_history or 0, max(0, num_frames - 1))

    cmd = build_command(
        script, os.path.join(MODEL_DIR, model_name),
        resolution, aspect_ratio, num_frames, fps, prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        # core args
        image=img_path,
        guidance_scale=(None if is_df else guidance_scale),
        shift=(None if is_df else shift),
        steps=steps,  # Acesta va fi mapat la inference_steps în build_command
        seed=seed,
        teacache_thresh=(teacache_thresh if not is_df else None),
        # DF-only args
        ar_step=(ar_step if is_df else None),
        base_num_frames=(base_num_frames if is_df else None),
        overlap_history=(overlap_history if is_df else None),
        addnoise_condition=(addnoise_condition if is_df else None),
        # flags
        offload=offload,
        teacache=(teacache if not is_df else None),
        use_ret_steps=(use_ret_steps if not is_df else None),
        use_usp=use_usp,
        cfg_rescale=cfg_rescale,
        prompt_enhancer=prompt_enhancer,
        causal_attention=causal_attention
    )

    logs = ""
    for line in run_and_yield_logs(cmd):
        logs += line + "\n"
        yield None, logs

    out = find_latest_video()
    yield out, logs

# -- MODEL DOWNLOAD UI -------------------------------------------------------

def create_download_interface():
    with gr.Blocks() as download_ui:
        gr.Markdown("### Download SkyReels-V2 Models")
        gr.Markdown("Select models to download:")
        
        missing_models = get_missing_models()
        
        if not missing_models:
            gr.Markdown("✅ All models are already downloaded.")
            return download_ui
        
        with gr.Group():
            # Creez o listă pentru a ține stările checkboxurilor
            checkbox_list = []
            
            # Adaug un checkbox pentru fiecare model lipsă
            for model in missing_models:
                desc = MODEL_INFO[model]["desc"]
                checkbox = gr.Checkbox(label=f"{model} - {desc}", value=False)
                checkbox_list.append(checkbox)
            
            with gr.Row():
                download_btn = gr.Button("Download Selected Models", variant="primary")
                select_all_btn = gr.Button("Select All")
                refresh_btn = gr.Button("Refresh Available Models", variant="secondary")
        
        download_progress = gr.Textbox(label="Download Progress", value="", interactive=False)
        download_status = gr.Textbox(label="Download Status", interactive=False)
        
        # Funcția pentru descărcarea modelelor selectate
        def download_models_callback(*checkbox_values):
            selected = [model for model, is_selected in zip(missing_models, checkbox_values) if is_selected]
            if not selected:
                return "Waiting for selection...", "No models selected."
            
            def update_progress(progress):
                return progress, download_status.value
            
            result = download_selected_models(selected, lambda p: update_progress(p))
            return "Download complete!", result
        
        # Conectez toate checkboxurile la butonul de descărcare
        download_btn.click(
            download_models_callback,
            inputs=checkbox_list,
            outputs=[download_progress, download_status]
        )
        
        # Funcția pentru selectarea tuturor modelelor
        def select_all():
            return [gr.update(value=True) for _ in checkbox_list]
        
        # Conectez butonul "Select All" la toate checkboxurile
        select_all_btn.click(
            select_all,
            outputs=checkbox_list
        )
        
        # Funcția pentru reîmprospătarea listei de modele
        def refresh_models():
            get_available_models()  # Refresh the available models
            return "Models refreshed! You can now proceed with generation."
        
        refresh_btn.click(
            refresh_models,
            outputs=download_status
        )
    
    return download_ui

# -- CUSTOM THEME ------------------------------------------------------------

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray"
).set(
    body_text_color="#303030",
    button_primary_background_fill="#1f77b4",
    button_primary_background_fill_hover="#5599cc",
    button_primary_text_color="white",
    button_secondary_background_fill="#e0e0e0",
    button_secondary_background_fill_hover="#c0c0c0",
    button_secondary_text_color="#303030",
    checkbox_background_color="#1f77b4",
    checkbox_background_color_selected="#1f77b4",
    checkbox_border_color="#1f77b4",
    checkbox_border_color_focus="#3b97e4",
    checkbox_border_color_hover="#5599cc",
    checkbox_border_color_selected="#1f77b4",
    checkbox_label_background_fill="white",
    checkbox_label_background_fill_hover="#f0f0f0",
    checkbox_label_background_fill_selected="#e0e0e0",
    checkbox_label_text_color="#303030",
    checkbox_label_text_color_selected="#303030",
    slider_color="#1f77b4",
    slider_color_dark="#0d5b96"
)

# -- MAIN GRADIO UI ----------------------------------------------------------

def create_interface():
    # Get available models
    model_map, t2v_models, i2v_models = get_available_models()
    missing_models = get_missing_models()
    
    with gr.Blocks(title="SkyReels-V2 Demo", theme=custom_theme, css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .model-select { min-width: 300px; }
        .aspect-ratio-group { display: flex; align-items: center; }
        .aspect-ratio-selector { max-width: 150px; margin-right: 10px; }
        .info-text { font-size: 0.9em; color: #555; font-style: italic; }
        .header-gradient { 
            background: linear-gradient(90deg, #1f77b4, #5599cc);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-gradient h1 { 
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }
        .header-gradient p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .duration-info {
            padding: 5px 10px;
            background-color: #e9f7fe;
            border-radius: 4px;
            font-size: 0.9em;
            margin: 5px 0;
            border-left: 2px solid #1f77b4;
        }
    """) as demo:
        with gr.Row(elem_classes=["header-gradient"]):
            gr.Markdown("# SkyReels-V2 Video Generator")
            gr.Markdown("Create high-quality videos from text or images using your local SkyReels-V2 models")
        
        refresh_models_btn = gr.Button("Refresh Available Models", variant="secondary", size="sm")
        refresh_status = gr.Markdown("")
        
        # Show download UI if models are missing
        if missing_models:
            with gr.Group(elem_classes=["warning-box"]):  # Folosim Group în loc de Box
                gr.Markdown(f"⚠️ {len(missing_models)} models are not downloaded yet.")
            download_ui = create_download_interface()
        else:
            with gr.Group(elem_classes=["success-box"]):  # Folosim Group în loc de Box
                gr.Markdown("✅ All models are already downloaded.")
        
        # Main tabs for generation
        with gr.Tabs() as tabs:
            # Prompt Enhancer tab
            with gr.TabItem("Prompt Enhancer", id="prompt-enhancer"):
                with gr.Row():
                    with gr.Column():
                        enhance_input = gr.Textbox(
                            label="Original Prompt",
                            placeholder="Enter a basic prompt to enhance...",
                            lines=3
                        )
                        enhance_button = gr.Button("Enhance Prompt", variant="primary")
                        gr.Markdown("""
                        *Prompt Enhancer uses Qwen2.5-32B-Instruct to transform your basic prompts into detailed video generation captions.
                        It works best with short inputs, adding shot details, scene composition, lighting, atmosphere, and more.*
                        """, elem_classes=["info-text"])
                        
                    with gr.Column():
                        enhance_output = gr.Textbox(
                            label="Enhanced Prompt",
                            lines=8,
                            interactive=True
                        )
                        enhance_status = gr.Markdown("")
                
                enhance_button.click(
                    fn=run_prompt_enhancer,
                    inputs=[enhance_input],
                    outputs=[enhance_output]
                )
                
            # Text to Video tab
            with gr.TabItem("Text to Video", id="text-to-video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        if t2v_models:
                            t2v_model = gr.Dropdown(
                                t2v_models, 
                                label="Model", 
                                value=t2v_models[0] if t2v_models else None
                            )
                            gr.Markdown("*Select the text-to-video model to use for generation*", elem_classes=["info-text"])
                        else:
                            gr.Markdown("⚠️ No Text-to-Video models available. Please download models first.")
                            t2v_model = gr.Dropdown([], label="Model (no models available)")
                        
                        t2v_prompt = gr.Textbox(
                            lines=3, 
                            placeholder="Enter a detailed description of the video you want to generate...", 
                            label="Prompt"
                        )
                        t2v_enhance_btn = gr.Button("Enhance with Mistral-7B", variant="secondary", size="sm")
                        t2v_enhance_status = gr.Markdown("", elem_classes=["info-text"])
                        t2v_enhance_btn.click(
                            fn=enhance_prompt_with_mistral,
                            inputs=[t2v_prompt],
                            outputs=[t2v_prompt, t2v_enhance_status]
                        )
                        gr.Markdown("*A detailed prompt will lead to better results. Describe the scene, subjects, actions, and atmosphere.*", elem_classes=["info-text"])
                        
                        # Adăugăm negative prompt
                        t2v_negative_prompt = gr.Textbox(
                            lines=2,
                            label="Negative Prompt (optional)",
                            placeholder="Enter concepts to avoid in the video...",
                            value=ENGLISH_NEGATIVE_PROMPT
                        )
                        gr.Markdown("*Describe what you don't want to see in the video. Leave empty to use the default negative prompt.*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            t2v_resolution = gr.Dropdown(
                                ["540P", "720P"], 
                                label="Resolution", 
                                value="540P"
                            )
                            
                            t2v_aspect = gr.Dropdown(
                                list(ASPECT_RATIOS.keys()),
                                label="Aspect Ratio",
                                value="16:9 (Landscape)"
                            )
                        gr.Markdown("*540P is faster, 720P gives higher quality but needs more VRAM (40GB+)*", elem_classes=["info-text"])
                        
                        # Mărim limita pentru numărul de cadre și adăugăm estimarea duratei
                        t2v_num = gr.Slider(
                            16, 1800,  # Extins la 1800 de cadre (cca. 1-2 minute la 24 FPS)
                            label="Frames", 
                            value=48
                        )
                        t2v_fps = gr.Slider(
                            1, 60, 
                            label="FPS", 
                            value=24
                        )
                        t2v_duration_info = gr.Markdown("", elem_classes=["duration-info"])
                        
                        # Funcția pentru actualizarea informațiilor despre durată
                        def update_t2v_duration(frames, fps):
                            seconds = frames / fps
                            if seconds < 60:
                                return f"Video duration estimate: **{seconds:.1f}** seconds"
                            else:
                                minutes = seconds // 60
                                rem_seconds = seconds % 60
                                return f"Video duration estimate: **{minutes:.0f}m {rem_seconds:.0f}s** ({seconds:.1f} seconds total)"
                        
                        # Conectăm funcția la sliderele pentru cadre și FPS
                        t2v_num.change(
                            fn=update_t2v_duration,
                            inputs=[t2v_num, t2v_fps],
                            outputs=[t2v_duration_info]
                        )
                        t2v_fps.change(
                            fn=update_t2v_duration,
                            inputs=[t2v_num, t2v_fps],
                            outputs=[t2v_duration_info]
                        )
                        
                        gr.Markdown("*More frames = longer video. 24 FPS is standard film framerate.*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            t2v_guidance = gr.Slider(
                                1.0, 10.0, 
                                label="Guidance Scale", 
                                value=6.0
                            )
                            t2v_shift = gr.Slider(
                                1.0, 10.0, 
                                label="Shift", 
                                value=8.0
                            )
                        gr.Markdown("*For T2V: Guidance Scale 6.0 and Shift 8.0 recommended*", elem_classes=["info-text"])
                        
                        # Adăugăm controlul pentru steps (Inference Steps)
                        t2v_steps = gr.Slider(
                            1, 100, 
                            label="Inference Steps", 
                            value=30,
                            step=1
                        )
                        gr.Markdown("*Higher steps = better quality but slower generation. 30 steps recommended.*", elem_classes=["info-text"])
                        
                        t2v_seed = gr.Number(
                            label="Seed (optional)", 
                            value=None, 
                            precision=0
                        )
                        gr.Markdown("*Set for reproducible results, leave empty for random*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            t2v_offload = gr.Checkbox(
                                label="Offload", 
                                value=True
                            )
                            t2v_use_usp = gr.Checkbox(
                                label="Use USP", 
                                value=False
                            )
                        gr.Markdown("*Offload reduces VRAM usage. USP enables multi-GPU acceleration if available.*", elem_classes=["info-text"])
                        
                        with gr.Accordion("Advanced Quality Options", open=False):
                            t2v_cfg_rescale = gr.Checkbox(
                                label="CFG Rescale", 
                                value=False
                            )
                            t2v_prompt_enhancer = gr.Checkbox(
                                label="Prompt Enhancer", 
                                value=False
                            )
                            t2v_causal_attention = gr.Checkbox(
                                label="Causal Attention", 
                                value=False
                            )
                        gr.Markdown("*Advanced parameters that can improve visual quality in some cases*", elem_classes=["info-text"])
                        
                        with gr.Accordion("Diffusion Forcing Options", open=False):
                            t2v_ar_step = gr.Number(
                                label="AR Step", 
                                value=0, 
                                precision=0
                            )
                            gr.Markdown("*0 for synchronous, 5 for asynchronous generation*", elem_classes=["info-text"])
                            
                            t2v_base_nf = gr.Slider(
                                16, 1800,  # Extins la 1800 pentru consistență
                                label="Base Frames", 
                                value=48
                            )
                            gr.Markdown("*Base frame count (reduce to save VRAM)*", elem_classes=["info-text"])
                            
                            t2v_overlap = gr.Slider(
                                0, 60, 
                                label="Overlap History", 
                                value=17
                            )
                            gr.Markdown("*Overlap frames for long videos (17 or 37 recommended)*", elem_classes=["info-text"])
                            
                            t2v_noise_c = gr.Slider(
                                0, 60, 
                                label="Noise Condition", 
                                value=20
                            )
                            gr.Markdown("*Smooths long videos (20 recommended, max 50)*", elem_classes=["info-text"])
                        
                        t2v_button = gr.Button("Generate Video", variant="primary")
                    
                    with gr.Column(scale=2):
                        t2v_output = gr.Video(label="Generated Video")
                        t2v_logs = gr.Textbox(label="Live Logs", lines=15, interactive=False)
                
            # Image to Video tab
            with gr.TabItem("Image to Video", id="image-to-video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        if i2v_models:
                            i2v_model = gr.Dropdown(
                                i2v_models, 
                                label="Model", 
                                value=i2v_models[0] if i2v_models else None
                            )
                            gr.Markdown("*Select the image-to-video model to use for generation*", elem_classes=["info-text"])
                        else:
                            gr.Markdown("⚠️ No Image-to-Video models available. Please download models first.")
                            i2v_model = gr.Dropdown([], label="Model (no models available)")
                        
                        i2v_image = gr.Image(
                            type="pil", 
                            label="Input Image"
                        )
                        gr.Markdown("*Upload an image to animate*", elem_classes=["info-text"])
                        
                        i2v_prompt = gr.Textbox(
                            lines=3, 
                            placeholder="Describe how the image should animate into a video...", 
                            label="Prompt"
                        )
                        i2v_enhance_btn = gr.Button("Enhance with Mistral-7B", variant="secondary", size="sm")
                        i2v_enhance_status = gr.Markdown("", elem_classes=["info-text"])
                        i2v_enhance_btn.click(
                            fn=enhance_prompt_with_mistral,
                            inputs=[i2v_prompt],
                            outputs=[i2v_prompt, i2v_enhance_status]
                        )
                        gr.Markdown("*Describe the motion and action you want to see in the video*", elem_classes=["info-text"])
                        
                        # Adăugăm negative prompt
                        i2v_negative_prompt = gr.Textbox(
                            lines=2,
                            label="Negative Prompt (optional)",
                            placeholder="Enter concepts to avoid in the video...",
                            value=ENGLISH_NEGATIVE_PROMPT
                        )
                        gr.Markdown("*Describe what you don't want to see in the video. Leave empty to use the default negative prompt.*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            i2v_resolution = gr.Dropdown(
                                ["540P", "720P"], 
                                label="Resolution", 
                                value="720P"
                            )
                            i2v_aspect = gr.Dropdown(
                                list(ASPECT_RATIOS.keys()),
                                label="Aspect Ratio",
                                value="16:9 (Landscape)"
                            )
                        gr.Markdown("*540P is faster, 720P gives higher quality but needs more VRAM*", elem_classes=["info-text"])
                        
                        # Mărim limita pentru numărul de cadre și adăugăm estimarea duratei
                        i2v_num = gr.Slider(
                            16, 1800,  # Extins la 1800 de cadre
                            label="Frames", 
                            value=48
                        )
                        i2v_fps = gr.Slider(
                            1, 60, 
                            label="FPS", 
                            value=24
                        )
                        i2v_duration_info = gr.Markdown("", elem_classes=["duration-info"])
                        
                        # Funcția pentru actualizarea informațiilor despre durată
                        def update_i2v_duration(frames, fps):
                            seconds = frames / fps
                            if seconds < 60:
                                return f"Video duration estimate: **{seconds:.1f}** seconds"
                            else:
                                minutes = seconds // 60
                                rem_seconds = seconds % 60
                                return f"Video duration estimate: **{minutes:.0f}m {rem_seconds:.0f}s** ({seconds:.1f} seconds total)"
                        
                        # Conectăm funcția la sliderele pentru cadre și FPS
                        i2v_num.change(
                            fn=update_i2v_duration,
                            inputs=[i2v_num, i2v_fps],
                            outputs=[i2v_duration_info]
                        )
                        i2v_fps.change(
                            fn=update_i2v_duration,
                            inputs=[i2v_num, i2v_fps],
                            outputs=[i2v_duration_info]
                        )
                        
                        gr.Markdown("*More frames = longer video. 24 FPS is standard film framerate.*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            i2v_guidance = gr.Slider(
                                1.0, 10.0, 
                                label="Guidance Scale", 
                                value=5.0
                            )
                            i2v_shift = gr.Slider(
                                1.0, 10.0, 
                                label="Shift", 
                                value=3.0
                            )
                        gr.Markdown("*For I2V: Guidance Scale 5.0 and Shift 3.0 recommended*", elem_classes=["info-text"])
                        
                        # Adăugăm controlul pentru steps (mapat la inference_steps)
                        i2v_steps = gr.Slider(
                            1, 100, 
                            label="Inference Steps", 
                            value=30,
                            step=1
                        )
                        gr.Markdown("*Higher steps = better quality but slower generation. 30 steps recommended.*", elem_classes=["info-text"])
                        
                        i2v_seed = gr.Number(
                            label="Seed (optional)", 
                            value=None, 
                            precision=0
                        )
                        gr.Markdown("*Set for reproducible results, leave empty for random*", elem_classes=["info-text"])
                        
                        with gr.Row():
                            i2v_offload = gr.Checkbox(
                                label="Offload", 
                                value=True
                            )
                            i2v_use_usp = gr.Checkbox(
                                label="Use USP", 
                                value=False
                            )
                        gr.Markdown("*Offload reduces VRAM usage. USP enables multi-GPU acceleration if available.*", elem_classes=["info-text"])
                        
                        with gr.Accordion("TEACache Options", open=False):
                            i2v_teacache = gr.Checkbox(
                                label="Use TEACache", 
                                value=True
                            )
                            i2v_ret = gr.Checkbox(
                                label="Use RetSteps", 
                                value=True
                            )
                            i2v_tc_thresh = gr.Slider(
                                0.0, 1.0, 
                                label="TEACache Threshold", 
                                value=0.3
                            )
                        gr.Markdown("*TEACache speeds up inference. RetSteps improves speed and quality.*", elem_classes=["info-text"])
                        
                        with gr.Accordion("Advanced Quality Options", open=False):
                            i2v_cfg_rescale = gr.Checkbox(
                                label="CFG Rescale", 
                                value=False
                            )
                            i2v_prompt_enhancer = gr.Checkbox(
                                label="Prompt Enhancer", 
                                value=False
                            )
                            i2v_causal_attention = gr.Checkbox(
                                label="Causal Attention", 
                                value=False
                            )
                        gr.Markdown("*Advanced parameters that can improve visual quality in some cases*", elem_classes=["info-text"])

                        with gr.Accordion("Diffusion Forcing Options", open=False):
                            i2v_ar_step = gr.Number(
                                label="AR Step", 
                                value=0, 
                                precision=0
                            )
                            gr.Markdown("*0 for synchronous, 5 for asynchronous generation*", elem_classes=["info-text"])
                            
                            i2v_base_nf = gr.Slider(
                                16, 1800,
                                label="Base Frames", 
                                value=48
                            )
                            gr.Markdown("*Base frame count (reduce to save VRAM)*", elem_classes=["info-text"])
                            
                            i2v_overlap = gr.Slider(
                                0, 60, 
                                label="Overlap History", 
                                value=17
                            )
                            gr.Markdown("*Overlap frames for long videos (17 or 37 recommended)*", elem_classes=["info-text"])
                            
                            i2v_noise_c = gr.Slider(
                                0, 60, 
                                label="Noise Condition", 
                                value=20
                            )
                            gr.Markdown("*Smooths long videos (20 recommended, max 50)*", elem_classes=["info-text"])
                                                
                        i2v_button = gr.Button("Generate Video", variant="primary")
                    
                    with gr.Column(scale=2):
                        i2v_output = gr.Video(label="Generated Video")
                        i2v_logs = gr.Textbox(label="Live Logs", lines=15, interactive=False)
            
            # Help tab with documentation
            with gr.TabItem("Help & Tips", id="help-tips"):
                gr.Markdown("""
                # SkyReels-V2 Generator Help
                
                ## Models
                
                - **T2V (Text-to-Video)**: Generate videos directly from text prompts
                - **I2V (Image-to-Video)**: Animate a still image into a video
                - **DF (Diffusion Forcing)**: Models for generating longer videos with better consistency
                
                ## Resolution & Performance
                
                - **540P**: Lower resolution, faster generation, less VRAM required (works with 16GB+ VRAM)
                - **720P**: Higher resolution, better quality, requires more VRAM (24GB+ recommended, 40GB+ for 14B models)
                
                ## Key Parameters
                
                ### Text-to-Video
                - **Prompt**: Detailed descriptions work best. Describe subjects, actions, environment, lighting.
                - **Negative Prompt**: Concepts to avoid in generation. Use this to prevent unwanted elements.
                - **Guidance Scale**: 6.0 recommended for T2V. Higher values follow the prompt more strictly.
                - **Shift**: 8.0 recommended for T2V.
                - **Inference Steps**: 30 is a good balance between quality and speed. Increase for better quality.
                
                ### Image-to-Video
                - **Prompt**: Describe how the image should animate.
                - **Negative Prompt**: Concepts to avoid in generation. Use this to prevent unwanted elements.
                - **Guidance Scale**: 5.0 recommended for I2V.
                - **Shift**: 3.0 recommended for I2V.
                - **Inference Steps**: 30 is a good balance between quality and speed. Increase for better quality.
                
                ### Diffusion Forcing (DF)
                - **AR Step**: Set to 0 for synchronous generation, 5 for asynchronous.
                - **Base Frames**: Reduce to save VRAM, but may impact quality.
                - **Overlap History**: Use 17 or 37 for long videos.
                - **Noise Condition**: Set to 20 for smoother long videos (max 50).
                
                ## Long Video Generation Tips
                
                - For videos longer than 30 seconds, use Diffusion Forcing (DF) models
                - Set appropriate Overlap History (17 or 37 recommended)
                - Monitor your VRAM usage - longer videos require more memory
                - Consider lowering resolution to 540P for very long videos
                
                ## Optimization
                
                - **Offload**: Enable to reduce VRAM usage (may slow down generation slightly)
                - **TEACache**: Enable for faster inference
                - **RetSteps**: Enable with TEACache for better quality
                - **USP**: Enable for multi-GPU acceleration (if available)
                
                ## Advanced Quality Options
                
                - **CFG Rescale**: Can improve contrast and color accuracy
                - **Prompt Enhancer**: Enhances prompt with additional details
                - **Causal Attention**: Improves temporal consistency in videos
                
                ## Tips for Better Results
                
                1. Use detailed prompts that describe the scene, characters, actions, and atmosphere
                2. For longer videos, use DF models with appropriate overlap settings
                3. Try different seeds to get varied results
                4. Start with recommended parameter values before experimenting
                5. If your GPU has sufficient VRAM, use higher resolution and step counts
                6. For Image-to-Video, use clear images with good lighting and framing
                7. Use the Prompt Enhancer tab to create more detailed prompts from simple ideas
                """)
        
        # Function to refresh available models
        def refresh_available_models():
            model_map, new_t2v_models, new_i2v_models = get_available_models()
            
            # Update dropdown values
            t2v_model_update = gr.update(choices=new_t2v_models, value=new_t2v_models[0] if new_t2v_models else None)
            i2v_model_update = gr.update(choices=new_i2v_models, value=new_i2v_models[0] if new_i2v_models else None)
            
            refresh_msg = f"✅ Models refreshed successfully! Found {len(new_t2v_models)} T2V models and {len(new_i2v_models)} I2V models."
            return t2v_model_update, i2v_model_update, refresh_msg
        
        # Connect refresh button
        refresh_models_btn.click(
            fn=refresh_available_models,
            inputs=[],
            outputs=[t2v_model, i2v_model, refresh_status]
        )
        
        # Inițializare imediată a estimărilor de durată
        t2v_duration_info.value = update_t2v_duration(48, 24)
        i2v_duration_info.value = update_t2v_duration(48, 24)
        
        # Hook up callbacks
        t2v_button.click(
            fn=text_to_video_live,
            inputs=[
                t2v_prompt, t2v_negative_prompt, t2v_model, t2v_resolution, t2v_aspect, t2v_num, t2v_fps,
                t2v_guidance, t2v_shift, t2v_steps, t2v_seed, t2v_offload, t2v_use_usp,
                t2v_cfg_rescale, t2v_prompt_enhancer, t2v_causal_attention,
                t2v_ar_step, t2v_base_nf, t2v_overlap, t2v_noise_c
            ],
            outputs=[t2v_output, t2v_logs],
            show_progress=True
        )
        
        i2v_button.click(
            fn=image_to_video_live,
            inputs=[
                i2v_image, i2v_prompt, i2v_negative_prompt, i2v_model, i2v_resolution, i2v_aspect, i2v_num, i2v_fps,
                i2v_guidance, i2v_shift, i2v_steps, i2v_seed, i2v_offload, i2v_teacache,
                i2v_ret, i2v_tc_thresh, i2v_use_usp, i2v_cfg_rescale, i2v_prompt_enhancer,
                i2v_causal_attention,
                # DF model parameters
                i2v_ar_step, i2v_base_nf, i2v_overlap, i2v_noise_c
            ],
            outputs=[i2v_output, i2v_logs],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    # Asigură-te că directorul pentru modele există
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    demo = create_interface()
    demo.queue()
    demo.launch(share=True, debug=True)
