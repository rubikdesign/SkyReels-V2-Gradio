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

def _run_and_capture(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    logs = []
    for line in iter(proc.stdout.readline, ""):
        logs.append(line.rstrip())
    proc.wait()
    return proc.returncode, logs

def _find_latest_video():
    mp4s = glob(os.path.join(VIDEO_OUTPUT_DIR, "*.mp4"))
    return max(mp4s, key=os.path.getctime) if mp4s else None

# -- GENERATION FUNCTIONS -----------------------------------------------------

def text_to_video(
    prompt,
    model_name,
    resolution, num_frames, fps,
    guidance_scale, shift,
    seed, offload, use_usp,
    # DF-only
    ar_step, base_num_frames, overlap_history, addnoise_condition
):
    model_path = MODEL_MAP[model_name]
    is_df = model_name.startswith("SkyReels-V2-DF-")
    script = "generate_video_df.py" if is_df else "generate_video.py"

    # Clamp DF parameters so they never exceed num_frames
    if is_df:
        base_num_frames = min(base_num_frames, num_frames)
        overlap_history = min(overlap_history, max(0, num_frames - 1))

    cmd = [
        sys.executable, script,
        "--model_id", model_path,
        "--resolution", resolution,
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--prompt", prompt,
    ]
    if not is_df:
        cmd += ["--guidance_scale", str(guidance_scale), "--shift", str(shift)]
    else:
        cmd += [
            "--ar_step", str(ar_step),
            "--base_num_frames", str(base_num_frames),
            "--overlap_history", str(overlap_history),
            "--addnoise_condition", str(addnoise_condition),
        ]

    if offload:   cmd.append("--offload")
    if use_usp:   cmd.append("--use_usp")
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    start = time.time()
    ret, logs = _run_and_capture(cmd)
    elapsed = time.time() - start

    if ret != 0:
        return None, f"❌ Generation failed (exit {ret})\n" + "\n".join(logs[-10:])
    out = _find_latest_video()
    msg = f"✅ Done in {elapsed:.1f}s — {os.path.basename(out)}" if out else "❌ No video found"
    return out, msg

def image_to_video(
    image, prompt, model_name,
    resolution, num_frames, fps,
    guidance_scale, shift,
    seed, offload, teacache, use_ret_steps, teacache_thresh, use_usp
):
    img_path = os.path.join(IMAGE_OUTPUT_DIR, "temp_input.jpg")
    if isinstance(image, Image.Image):
        image.save(img_path)
    else:
        Image.fromarray(image).save(img_path)

    model_path = MODEL_MAP[model_name]
    cmd = [
        sys.executable, "generate_video.py",
        "--model_id", model_path,
        "--resolution", resolution,
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--image", img_path,
        "--prompt", prompt,
        "--guidance_scale", str(guidance_scale),
        "--shift", str(shift),
    ]
    if offload:       cmd.append("--offload")
    if teacache:      cmd.append("--teacache")
    if use_ret_steps: cmd.append("--use_ret_steps")
    if teacache_thresh is not None:
        cmd += ["--teacache_thresh", str(teacache_thresh)]
    if use_usp:       cmd.append("--use_usp")
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    start = time.time()
    ret, logs = _run_and_capture(cmd)
    elapsed = time.time() - start

    if ret != 0:
        return None, f"❌ Generation failed (exit {ret})\n" + "\n".join(logs[-10:])
    out = _find_latest_video()
    msg = f"✅ Done in {elapsed:.1f}s — {os.path.basename(out)}" if out else "❌ No video found"
    return out, msg

# -- GRADIO UI ---------------------------------------------------------------

with gr.Blocks(title="SkyReels-V2 Demo") as demo:
    gr.Markdown("## SkyReels-V2 Video Generator")
    gr.Markdown("**Text-to-Video** and **Image-to-Video** using your local SkyReels-V2 models")

    # — Text to Video —
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column(scale=2):
                t2v_model    = gr.Dropdown(T2V_MODELS, value=T2V_MODELS[0], label="Model")
                t2v_prompt   = gr.Textbox(lines=3, placeholder="Enter your video prompt…", label="Prompt")
                with gr.Row():
                    t2v_resolution = gr.Dropdown(["540P","720P"], value="540P", label="Resolution")
                    # Allow up to 3600 frames (~150s at 24fps)
                    t2v_num        = gr.Slider(1, 3600, value=48, step=1, label="Frames")
                    t2v_fps        = gr.Slider(1, 60,   value=24, step=1, label="FPS")
                with gr.Row():
                    t2v_guid  = gr.Slider(1.0, 10.0, value=6.0, step=0.1, label="Guidance Scale")
                    t2v_shift = gr.Slider(1.0, 10.0, value=8.0, step=0.1, label="Shift")
                t2v_seed    = gr.Number(value=None, precision=0, label="Seed (optional)")
                t2v_offload = gr.Checkbox(value=True,  label="Offload")
                t2v_use_usp = gr.Checkbox(value=False, label="Use USPosing (--use_usp)")
                with gr.Accordion("DF-Only Advanced Options", open=False):
                    ar_step   = gr.Number(value=0,    precision=0, label="ar_step")
                    # Match the same upper bound as num_frames
                    base_nf   = gr.Slider(1, 3600,    value=48,  step=1, label="base_num_frames")
                    overlap   = gr.Slider(0, 3599,    value=17,  step=1, label="overlap_history")
                    noise_c   = gr.Slider(0, 100,     value=20,  step=1, label="addnoise_condition")
                t2v_button = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=2):
                t2v_out = gr.Video(label="Generated Video")
                t2v_msg = gr.Textbox(label="Status", interactive=False)

    # — Image to Video —
    with gr.Tab("Image to Video"):
        with gr.Row():
            with gr.Column(scale=2):
                i2v_model     = gr.Dropdown(I2V_MODELS, value=I2V_MODELS[0], label="Model")
                i2v_image     = gr.Image(type="pil", label="Input Image")
                i2v_prompt    = gr.Textbox(lines=2, placeholder="Enter a short prompt…", label="Prompt")
                with gr.Row():
                    i2v_resolution = gr.Dropdown(["540P","720P"], value="720P", label="Resolution")
                    i2v_num        = gr.Slider(1, 3600, value=48, step=1, label="Frames")
                    i2v_fps        = gr.Slider(1, 60,   value=24, step=1, label="FPS")
                with gr.Row():
                    i2v_guid  = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Guidance Scale")
                    i2v_shift = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="Shift")
                i2v_seed      = gr.Number(value=None, precision=0, label="Seed (optional)")
                i2v_offload   = gr.Checkbox(value=True,  label="Offload")
                i2v_teacache  = gr.Checkbox(value=True,  label="Use TEACache")
                i2v_ret       = gr.Checkbox(value=True,  label="Use RetSteps")
                i2v_tc_thresh = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="TEACache Threshold")
                i2v_use_usp   = gr.Checkbox(value=False, label="Use USPosing (--use_usp)")
                i2v_button    = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=2):
                i2v_out = gr.Video(label="Generated Video")
                i2v_msg = gr.Textbox(label="Status", interactive=False)

    # — Hook up callbacks —
    t2v_button.click(
        fn=text_to_video,
        inputs=[
            t2v_prompt, t2v_model,
            t2v_resolution, t2v_num, t2v_fps,
            t2v_guid, t2v_shift,
            t2v_seed, t2v_offload, t2v_use_usp,
            ar_step, base_nf, overlap, noise_c
        ],
        outputs=[t2v_out, t2v_msg],
        show_progress=True
    )
    i2v_button.click(
        fn=image_to_video,
        inputs=[
            i2v_image, i2v_prompt, i2v_model,
            i2v_resolution, i2v_num, i2v_fps,
            i2v_guid, i2v_shift,
            i2v_seed, i2v_offload, i2v_teacache,
            i2v_ret, i2v_tc_thresh, i2v_use_usp
        ],
        outputs=[i2v_out, i2v_msg],
        show_progress=True
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, debug=True)
