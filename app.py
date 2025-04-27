#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from PIL import Image

# -- CONFIG -------------------------------------------------------------------

MODEL_DIR = "./models"
VIDEO_OUTPUT_DIR = "./result/video_out"
IMAGE_OUTPUT_DIR = "./image_out"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Discover all local models under ./models
all_dirs = sorted(glob(os.path.join(MODEL_DIR, "SkyReels-V2-*")))
MODEL_MAP = {os.path.basename(p): p for p in all_dirs}

# Split into T2V+DF vs I2V sets
T2V_MODELS = [m for m in MODEL_MAP if "-T2V-" in m] + [m for m in MODEL_MAP if "-DF-" in m]
I2V_MODELS = [m for m in MODEL_MAP if "-I2V-" in m]

# -- HELPERS ------------------------------------------------------------------

def run_and_capture(cmd):
    """Run a command and capture its output."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    logs = []
    for line in iter(proc.stdout.readline, ""):
        logs.append(line.rstrip())
    proc.wait()
    return proc.returncode, logs

def find_latest_video():
    """Find the latest generated video file."""
    mp4s = glob(os.path.join(VIDEO_OUTPUT_DIR, "*.mp4"))
    return max(mp4s, key=os.path.getctime) if mp4s else None

def build_command(script, model_path, resolution, num_frames, fps, prompt, **kwargs):
    """Build a command list for subprocess execution."""
    cmd = [
        sys.executable, script,
        "--model_id", model_path,
        "--resolution", resolution,
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--prompt", prompt,
    ]
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    return cmd

# -- GENERATION FUNCTIONS -----------------------------------------------------

def text_to_video(
    prompt, model_name, resolution, num_frames, fps,
    guidance_scale, shift, seed, offload, use_usp,
    ar_step=None, base_num_frames=None, overlap_history=None, addnoise_condition=None
):
    """Generate a video from text input."""
    model_path = MODEL_MAP[model_name]
    is_df = model_name.startswith("SkyReels-V2-DF-")
    script = "generate_video_df.py" if is_df else "generate_video.py"

    # Clamp DF parameters so they never exceed num_frames
    if is_df:
        base_num_frames = min(base_num_frames or num_frames, num_frames)
        overlap_history = min(overlap_history or 0, max(0, num_frames - 1))

    cmd = build_command(
        script, model_path, resolution, num_frames, fps, prompt,
        guidance_scale=guidance_scale, shift=shift, seed=seed,
        offload=offload, use_usp=use_usp,
        ar_step=ar_step, base_num_frames=base_num_frames,
        overlap_history=overlap_history, addnoise_condition=addnoise_condition,
        outdir=VIDEO_OUTPUT_DIR
    )

    start = time.time()
    ret, logs = run_and_capture(cmd)
    elapsed = time.time() - start

    if ret != 0:
        return None, f"❌ Generation failed (exit {ret})\n" + "\n".join(logs[-10:])
    out = find_latest_video()
    msg = f"✅ Done in {elapsed:.1f}s — {os.path.basename(out)}" if out else "❌ No video found"
    return out, msg

def image_to_video(
    image, prompt, model_name, resolution, num_frames, fps,
    guidance_scale, shift, seed, offload, teacache, use_ret_steps, teacache_thresh, use_usp
):
    """Generate a video from an image input."""
    img_path = os.path.join(IMAGE_OUTPUT_DIR, "temp_input.jpg")
    if isinstance(image, Image.Image):
        image.save(img_path)
    else:
        Image.fromarray(image).save(img_path)

    model_path = MODEL_MAP[model_name]
    cmd = build_command(
        "generate_video.py", model_path, resolution, num_frames, fps, prompt,
        image=img_path, guidance_scale=guidance_scale, shift=shift, seed=seed,
        offload=offload, teacache=teacache, use_ret_steps=use_ret_steps,
        teacache_thresh=teacache_thresh, use_usp=use_usp,
        outdir=VIDEO_OUTPUT_DIR
    )

    start = time.time()
    ret, logs = run_and_capture(cmd)
    elapsed = time.time() - start

    if ret != 0:
        return None, f"❌ Generation failed (exit {ret})\n" + "\n".join(logs[-10:])
    out = find_latest_video()
    msg = f"✅ Done in {elapsed:.1f}s — {os.path.basename(out)}" if out else "❌ No video found"
    return out, msg

# -- GRADIO UI ---------------------------------------------------------------

with gr.Blocks(title="SkyReels-V2 Demo") as demo:
    gr.Markdown("# SkyReels-V2 Video Generator")
    gr.Markdown("Create videos from text or images using your local SkyReels-V2 models.")

    # -- Text to Video --
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column(scale=2):
                t2v_model = gr.Dropdown(T2V_MODELS, label="Model", value=T2V_MODELS[0])
                t2v_prompt = gr.Textbox(lines=3, placeholder="Enter your video prompt...", label="Prompt")
                with gr.Row():
                    t2v_resolution = gr.Dropdown(["540P", "720P"], label="Resolution", value="540P")
                    t2v_num_frames = gr.Slider(1, 3600, label="Frames", value=48)
                    t2v_fps = gr.Slider(1, 60, label="FPS", value=24)
                with gr.Row():
                    t2v_guidance = gr.Slider(1.0, 10.0, label="Guidance Scale", value=6.0)
                    t2v_shift = gr.Slider(1.0, 10.0, label="Shift", value=8.0)
                t2v_seed = gr.Number(label="Seed (optional)", value=None)
                t2v_offload = gr.Checkbox(label="Offload", value=True)
                t2v_use_usp = gr.Checkbox(label="Use USPosing (--use_usp)", value=False)
                with gr.Accordion("DF-Only Advanced Options", open=False):
                    t2v_ar_step = gr.Number(label="ar_step", value=0)
                    t2v_base_nf = gr.Slider(1, 3600, label="base_num_frames", value=48)
                    t2v_overlap = gr.Slider(0, 3599, label="overlap_history", value=17)
                    t2v_noise_c = gr.Slider(0, 100, label="addnoise_condition", value=20)
                t2v_button = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=2):
                t2v_output = gr.Video(label="Generated Video")
                t2v_status = gr.Textbox(label="Status", interactive=False)

    # -- Image to Video --
    with gr.Tab("Image to Video"):
        with gr.Row():
            with gr.Column(scale=2):
                i2v_model = gr.Dropdown(I2V_MODELS, label="Model", value=I2V_MODELS[0])
                i2v_image = gr.Image(type="pil", label="Input Image")
                i2v_prompt = gr.Textbox(lines=2, placeholder="Enter a short prompt...", label="Prompt")
                with gr.Row():
                    i2v_resolution = gr.Dropdown(["540P", "720P"], label="Resolution", value="720P")
                    i2v_num_frames = gr.Slider(1, 3600, label="Frames", value=48)
                    i2v_fps = gr.Slider(1, 60, label="FPS", value=24)
                with gr.Row():
                    i2v_guidance = gr.Slider(1.0, 10.0, label="Guidance Scale", value=5.0)
                    i2v_shift = gr.Slider(1.0, 10.0, label="Shift", value=3.0)
                i2v_seed = gr.Number(label="Seed (optional)", value=None)
                i2v_offload = gr.Checkbox(label="Offload", value=True)
                i2v_teacache = gr.Checkbox(label="Use TEACache", value=True)
                i2v_ret = gr.Checkbox(label="Use RetSteps", value=True)
                i2v_tc_thresh = gr.Slider(0.0, 1.0, label="TEACache Threshold", value=0.3)
                i2v_use_usp = gr.Checkbox(label="Use USPosing (--use_usp)", value=False)
                i2v_button = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=2):
                i2v_output = gr.Video(label="Generated Video")
                i2v_status = gr.Textbox(label="Status", interactive=False)

    # -- Hook up callbacks --
    t2v_button.click(
        fn=text_to_video,
        inputs=[
            t2v_prompt, t2v_model, t2v_resolution, t2v_num_frames, t2v_fps,
            t2v_guidance, t2v_shift, t2v_seed, t2v_offload, t2v_use_usp,
            t2v_ar_step, t2v_base_nf, t2v_overlap, t2v_noise_c
        ],
        outputs=[t2v_output, t2v_status],
        show_progress=True
    )
    i2v_button.click(
        fn=image_to_video,
        inputs=[
            i2v_image, i2v_prompt, i2v_model, i2v_resolution, i2v_num_frames, i2v_fps,
            i2v_guidance, i2v_shift, i2v_seed, i2v_offload, i2v_teacache,
            i2v_ret, i2v_tc_thresh, i2v_use_usp
        ],
        outputs=[i2v_output, i2v_status],
        show_progress=True
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, debug=True)
