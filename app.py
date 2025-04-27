import os
import gradio as gr
import argparse
import torch
import numpy as np
import subprocess
import time
import threading
from PIL import Image

# Check if Gradio is installed and install it if not
try:
    import gradio
except ImportError:
    print("Installing Gradio...")
    subprocess.run(["pip", "install", "gradio"], check=True)
    import gradio as gr

# Check if Diffusers is installed and install it if not
try:
    import diffusers
except ImportError:
    print("Installing Diffusers...")
    subprocess.run(["pip", "install", "diffusers"], check=True)
    import diffusers
    
# Configure output directories
VIDEO_OUTPUT_DIR = "./result/video_out"
IMAGE_OUTPUT_DIR = "./image_out"

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Function for text-to-image
def text_to_image(prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Use diffusers for text-to-image
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    # Save the image
    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"generated_image_{hash(prompt)}.png")
    image.save(image_path)
    
    return image, image_path

# Function for image-to-video
def image_to_video(image, prompt, resolution="720P", num_frames=121, guidance_scale=5.0, 
                   shift=3.0, fps=24, seed=None, offload=True, teacache=True, 
                   use_ret_steps=True, teacache_thresh=0.3):
    
    # Display progress in console
    current_status = "Initializing video generation process..."
    print(current_status)
    
    # Save the image temporarily
    temp_image_path = os.path.join(IMAGE_OUTPUT_DIR, "temp_input.jpg")
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_image_path)
    else:
        image.save(temp_image_path)
    
    current_status = "Image saved temporarily. Preparing for video generation..."
    print(current_status)
    
    # Build command for generate_video.py
    cmd = [
        "python3", "generate_video.py",
        "--model_id", "./models/SkyReels-V2-I2V-14B-720P",
        "--resolution", resolution,
        "--num_frames", str(num_frames),
        "--guidance_scale", str(guidance_scale),
        "--shift", str(shift),
        "--fps", str(fps),
        "--image", temp_image_path,
        "--prompt", prompt,
        "--outdir", "video_out"  # Make sure the output directory is correct
    ]
    
    if seed is not None and seed != 0:
        cmd.extend(["--seed", str(int(seed))])
    
    if offload:
        cmd.append("--offload")
    
    if teacache:
        cmd.append("--teacache")
    
    if use_ret_steps:
        cmd.append("--use_ret_steps")
    
    if teacache_thresh:
        cmd.extend(["--teacache_thresh", str(teacache_thresh)])
    
    current_status = f"Starting process with command:\n{' '.join(cmd)}"
    print(current_status)
    
    # Execute command with pipes for stdout and stderr
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        text=True
    )
    
    # Read output in real time
    logs = []
    
    # Read and update output in real time
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            logs.append(line.strip())
            print(line.strip())
    
    # Process has finished, check exit code
    process.wait()
    
    if process.returncode != 0:
        # Read any errors
        stderr = process.stderr.read()
        error_msg = f"Error during generation. Exit code: {process.returncode}\n{stderr}"
        print(error_msg)
        return None, error_msg
    
    current_status = "\nGeneration complete. Searching for video file..."
    print(current_status)
    
    # Find the last video recording created in the VIDEO_OUTPUT_DIR directory
    video_files = [os.path.join(VIDEO_OUTPUT_DIR, f) for f in os.listdir(VIDEO_OUTPUT_DIR) 
                  if f.endswith('.mp4')]
    
    if not video_files:
        error_msg = "No generated video file was found."
        print(error_msg)
        return None, error_msg
    
    video_path = max(video_files, key=os.path.getctime)
    success_msg = f"Video successfully generated at: {video_path}"
    print(success_msg)
    
    return video_path, success_msg

# Gradio Interface
def create_interface():
    with gr.Blocks(title="SkyReels-V2 Demo") as demo:
        gr.Markdown("# SkyReels-V2 Demo Generator")
        gr.Markdown("Generate videos from images or text using SkyReels-V2 models")
        
        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Describe the desired image", lines=3)
                    t2i_steps = gr.Slider(minimum=20, maximum=100, value=50, step=1, label="Inference Steps")
                    t2i_guidance = gr.Slider(minimum=1, maximum=15, value=7.5, step=0.1, label="Guidance Scale")
                    t2i_seed = gr.Number(label="Seed (optional)", precision=0)
                    t2i_button = gr.Button("Generate Image")
                
                with gr.Column():
                    t2i_output = gr.Image(label="Generated Image")
                    t2i_path = gr.Textbox(label="Path to the generated image")
        
        with gr.Tab("Image to Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    i2v_image = gr.Image(label="Upload an image or generate one from the Text to Image tab", type="pil")
                    i2v_prompt = gr.Textbox(label="Describe the desired video", lines=3)
                    
                    with gr.Row():
                        with gr.Column():
                            i2v_resolution = gr.Dropdown(choices=["540P", "720P"], value="720P", label="Resolution")
                            i2v_frames = gr.Slider(minimum=97, maximum=257, value=121, step=1, label="Number of Frames")
                            i2v_guidance = gr.Slider(minimum=1, maximum=10, value=5.0, step=0.1, label="Guidance Scale")
                            i2v_shift = gr.Slider(minimum=1, maximum=10, value=3.0, step=0.1, label="Shift")
                            i2v_fps = gr.Slider(minimum=15, maximum=60, value=24, step=1, label="FPS")
                            i2v_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        with gr.Column():
                            i2v_offload = gr.Checkbox(value=True, label="Offload")
                            i2v_teacache = gr.Checkbox(value=True, label="Use Teacache")
                            i2v_use_ret_steps = gr.Checkbox(value=True, label="Use Retention Steps")
                            i2v_teacache_thresh = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Teacache Threshold")
                    
                    i2v_button = gr.Button("Generate Video", variant="primary")
                
                with gr.Column(scale=2):
                    i2v_output = gr.Video(label="Generated Video")
                    i2v_message = gr.Textbox(label="Status Message")
        
        # Connect buttons to functions
        t2i_button.click(
            text_to_image, 
            inputs=[t2i_prompt, t2i_steps, t2i_guidance, t2i_seed],
            outputs=[t2i_output, t2i_path]
        )
        
        i2v_button.click(
            fn=image_to_video,
            inputs=[
                i2v_image, i2v_prompt, i2v_resolution, i2v_frames, i2v_guidance, 
                i2v_shift, i2v_fps, i2v_seed, i2v_offload, i2v_teacache, 
                i2v_use_ret_steps, i2v_teacache_thresh
            ],
            outputs=[i2v_output, i2v_message],
            show_progress=True
        )
        
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.queue()  # Activate the Gradio queue system
    interface.launch(share=True, debug=True)
