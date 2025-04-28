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

# -- CONFIG -------------------------------------------------------------------

MODEL_DIR = "./models"
VIDEO_OUTPUT_DIR = "./result/video_out"
IMAGE_OUTPUT_DIR = "./image_out"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

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
    all_dirs = sorted(glob(os.path.join(MODEL_DIR, "SkyReels-V2-*")))
    model_map = {os.path.basename(p): p for p in all_dirs}
    
    # Split into model types
    t2v_models = [m for m in model_map if "-T2V-" in m] + [m for m in model_map if "-DF-" in m]
    i2v_models = [m for m in model_map if "-I2V-" in m]
    
    return model_map, t2v_models, i2v_models

def get_missing_models():
    """Find which models are not downloaded yet"""
    model_map, _, _ = get_available_models()
    return [name for name in MODEL_INFO.keys() if name not in model_map]

def download_model_huggingface(model_name, progress=None):
    """Download a model from Hugging Face"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
    
    repo_id = MODEL_INFO[model_name]["repo_id"]
    local_dir = os.path.join(MODEL_DIR, model_name)
    
    print(f"Downloading {model_name}...")
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
        print(f"Successfully downloaded {model_name}")
        return f"✅ Successfully downloaded {model_name}"
    except Exception as e:
        return f"❌ Failed to download {model_name}: {str(e)}"

def download_selected_models(selected_models):
    """Download multiple selected models"""
    results = []
    total = len(selected_models)
    
    for i, model in enumerate(selected_models):
        print(f"Downloading {model} ({i+1}/{total})...")
        result = download_model_huggingface(model)
        results.append(result)
        print(f"Completed {model} ({i+1}/{total})")
    
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
    "prompt_enhancer"
}

def build_command(script, model_path, resolution, aspect_ratio, num_frames, fps, prompt, **kwargs):
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
    
    # Add the width and height parameters directly
    cmd.extend(["--width", str(width), "--height", str(height)])
    
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

# -- GENERATION FUNCTIONS -----------------------------------------------------

def text_to_video_live(
    prompt, model_name, resolution, aspect_ratio, num_frames, fps,
    guidance_scale, shift, seed, offload, use_usp,
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
        # core args
        guidance_scale=(None if is_df else guidance_scale),
        shift=(None if is_df else shift),
        seed=seed,
        # DF-only args
        ar_step=(ar_step if is_df else None),
        base_num_frames=(base_num_frames if is_df else None),
        overlap_history=(overlap_history if is_df else None),
        addnoise_condition=(addnoise_condition if is_df else None),
        # flags
        offload=offload,
        use_usp=use_usp
    )

    logs = ""
    for line in run_and_yield_logs(cmd):
        logs += line + "\n"
        yield None, logs

    out = find_latest_video()
    yield out, logs

def image_to_video_live(
    image, prompt, model_name, resolution, aspect_ratio, num_frames, fps,
    guidance_scale, shift, seed, offload, teacache, use_ret_steps, teacache_thresh, use_usp
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

    cmd = build_command(
        "generate_video.py", os.path.join(MODEL_DIR, model_name),
        resolution, aspect_ratio, num_frames, fps, prompt,
        # core args
        image=img_path,
        guidance_scale=guidance_scale,
        shift=shift,
        seed=seed,
        teacache_thresh=teacache_thresh,
        # flags
        offload=offload,
        teacache=teacache,
        use_ret_steps=use_ret_steps,
        use_usp=use_usp
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
        model_checkboxes = {}
        
        if not missing_models:
            gr.Markdown("✅ All models are already downloaded.")
        else:
            with gr.Group():
                for model in missing_models:
                    desc = MODEL_INFO[model]["desc"]
                    model_checkboxes[model] = gr.Checkbox(label=f"{model} - {desc}", value=False)
                
                with gr.Row():
                    download_btn = gr.Button("Download Selected Models", variant="primary")
                    select_all_btn = gr.Button("Select All")
            
            download_status = gr.Textbox(label="Download Status", interactive=False)
            
            # Download button callback
            def download_models_callback():
                selected = [model for model, checkbox in model_checkboxes.items() if checkbox.value]
                if not selected:
                    return "No models selected."
                return download_selected_models(selected)
            
            download_btn.click(download_models_callback, outputs=download_status)
            
            # Select all button callback
            def select_all():
                return [gr.update(value=True) for _ in model_checkboxes]
            
            select_all_btn.click(select_all, outputs=list(model_checkboxes.values()))
    
    return download_ui

# -- MAIN GRADIO UI ----------------------------------------------------------

def create_interface():
    # Get available models
    model_map, t2v_models, i2v_models = get_available_models()
    missing_models = get_missing_models()
    
    with gr.Blocks(title="SkyReels-V2 Demo", css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .model-select { min-width: 300px; }
        .aspect-ratio-group { display: flex; align-items: center; }
        .aspect-ratio-selector { max-width: 150px; margin-right: 10px; }
        .info-text { font-size: 0.9em; color: #666; font-style: italic; }
    """) as demo:
        gr.Markdown("# SkyReels-V2 Video Generator")
        gr.Markdown("Create videos from text or images using your local SkyReels-V2 models.")
        
        # Show download UI if models are missing
        if missing_models:
            gr.Markdown(f"⚠️ {len(missing_models)} models are not downloaded yet.")
            download_ui = create_download_interface()
        
        # Main tabs for generation
        with gr.Tab("Text to Video"):
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
                    gr.Markdown("*A detailed prompt will lead to better results. Describe the scene, subjects, actions, and atmosphere.*", elem_classes=["info-text"])
                    
                    with gr.Row():
                        t2v_resolution = gr.Dropdown(
                            ["540P", "720P"], 
                            label="Resolution", 
                            value="540P"
                        )
                        gr.Markdown("*540P is faster, 720P gives higher quality but needs more VRAM (40GB+)*", elem_classes=["info-text"])
                        
                        t2v_aspect = gr.Dropdown(
                            list(ASPECT_RATIOS.keys()),
                            label="Aspect Ratio",
                            value="16:9 (Landscape)"
                        )
                    
                    with gr.Row():
                        t2v_num = gr.Slider(
                            16, 360, 
                            label="Frames", 
                            value=48
                        )
                        t2v_fps = gr.Slider(
                            1, 60, 
                            label="FPS", 
                            value=24
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
                    
                    with gr.Accordion("Advanced Options (Diffusion Forcing)", open=False):
                        t2v_ar_step = gr.Number(
                            label="AR Step", 
                            value=0, 
                            precision=0
                        )
                        gr.Markdown("*0 for synchronous, 5 for asynchronous generation*", elem_classes=["info-text"])
                        
                        t2v_base_nf = gr.Slider(
                            16, 360, 
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
        with gr.Tab("Image to Video"):
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
                    gr.Markdown("*Describe the motion and action you want to see in the video*", elem_classes=["info-text"])
                    
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
                    
                    with gr.Row():
                        i2v_num = gr.Slider(
                            16, 360, 
                            label="Frames", 
                            value=48
                        )
                        i2v_fps = gr.Slider(
                            1, 60, 
                            label="FPS", 
                            value=24
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
                    
                    with gr.Accordion("Advanced Options", open=False):
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
                    
                    i2v_button = gr.Button("Generate Video", variant="primary")
                
                with gr.Column(scale=2):
                    i2v_output = gr.Video(label="Generated Video")
                    i2v_logs = gr.Textbox(label="Live Logs", lines=15, interactive=False)
        
        # Help tab with documentation
        with gr.Tab("Help & Tips"):
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
            - **Guidance Scale**: 6.0 recommended for T2V. Higher values follow the prompt more strictly.
            - **Shift**: 8.0 recommended for T2V.
            
            ### Image-to-Video
            - **Prompt**: Describe how the image should animate.
            - **Guidance Scale**: 5.0 recommended for I2V.
            - **Shift**: 3.0 recommended for I2V.
            
            ### Diffusion Forcing (DF)
            - **AR Step**: Set to 0 for synchronous generation, 5 for asynchronous.
            - **Base Frames**: Reduce to save VRAM, but may impact quality.
            - **Overlap History**: Use 17 or 37 for long videos.
            - **Noise Condition**: Set to 20 for smoother long videos (max 50).
            
            ## Optimization
            
            - **Offload**: Enable to reduce VRAM usage (may slow down generation slightly)
            - **TEACache**: Enable for faster inference
            - **RetSteps**: Enable with TEACache for better quality
            - **USP**: Enable for multi-GPU acceleration (if available)
            
            ## Tips for Better Results
            
            1. Use detailed prompts that describe the scene, characters, actions, and atmosphere
            2. For longer videos, use DF models with appropriate overlap settings
            3. Try different seeds to get varied results
            4. Start with recommended parameter values before experimenting
            """)
        
        # Hook up callbacks
        t2v_button.click(
            fn=text_to_video_live,
            inputs=[
                t2v_prompt, t2v_model, t2v_resolution, t2v_aspect, t2v_num, t2v_fps,
                t2v_guidance, t2v_shift, t2v_seed, t2v_offload, t2v_use_usp,
                t2v_ar_step, t2v_base_nf, t2v_overlap, t2v_noise_c
            ],
            outputs=[t2v_output, t2v_logs],
            show_progress=True
        )
        
        i2v_button.click(
            fn=image_to_video_live,
            inputs=[
                i2v_image, i2v_prompt, i2v_model, i2v_resolution, i2v_aspect, i2v_num, i2v_fps,
                i2v_guidance, i2v_shift, i2v_seed, i2v_offload, i2v_teacache,
                i2v_ret, i2v_tc_thresh, i2v_use_usp
            ],
            outputs=[i2v_output, i2v_logs],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(share=True, debug=True)
