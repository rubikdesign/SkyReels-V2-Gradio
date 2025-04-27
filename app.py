import os
import gradio as gr
import argparse
import torch
import numpy as np
import subprocess
import time
import threading
from PIL import Image

# Verifică dacă Gradio este instalat și instalează-l dacă nu este
try:
    import gradio
except ImportError:
    print("Instalare Gradio...")
    subprocess.run(["pip", "install", "gradio"], check=True)
    import gradio as gr

# Verifică dacă Diffusers este instalat și instalează-l dacă nu este
try:
    import diffusers
except ImportError:
    print("Instalare Diffusers...")
    subprocess.run(["pip", "install", "diffusers"], check=True)
    import diffusers
    
# Configurați directoarele de ieșire
VIDEO_OUTPUT_DIR = "./result/video_out"
IMAGE_OUTPUT_DIR = "./image_out"

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Funcția pentru text-to-image
def text_to_image(prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Folosim diffusers pentru text-to-image
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    # Salvează imaginea
    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"generated_image_{hash(prompt)}.png")
    image.save(image_path)
    
    return image, image_path

# Funcția pentru image-to-video
def image_to_video(image, prompt, resolution="720P", num_frames=121, guidance_scale=5.0, 
                   shift=3.0, fps=24, seed=None, offload=True, teacache=True, 
                   use_ret_steps=True, teacache_thresh=0.3):
    
    # Afișăm în consolă progresul
    current_status = "Inițializare proces de generare video..."
    print(current_status)
    
    # Salvează imaginea temporar
    temp_image_path = os.path.join(IMAGE_OUTPUT_DIR, "temp_input.jpg")
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_image_path)
    else:
        image.save(temp_image_path)
    
    current_status = "Imagine salvată temporar. Pregătire pentru generare video..."
    print(current_status)
    
    # Construiește comanda pentru generate_video.py
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
        "--outdir", "video_out"  # Asigură-te că directorul de ieșire este corect
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
    
    current_status = f"Pornire proces cu comanda:\n{' '.join(cmd)}"
    print(current_status)
    
    # Execută comanda cu pipe-uri pentru stdout și stderr
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        text=True
    )
    
    # Citim outputul în timp real
    logs = []
    
    # Citește și actualizează outputul în timp real
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            logs.append(line.strip())
            print(line.strip())
    
    # Procesul s-a terminat, verificăm codul de ieșire
    process.wait()
    
    if process.returncode != 0:
        # Citim și eventualele erori
        stderr = process.stderr.read()
        error_msg = f"Eroare în timpul generării. Cod de ieșire: {process.returncode}\n{stderr}"
        print(error_msg)
        return None, error_msg
    
    current_status = "\nGenerare completă. Căutare fișier video..."
    print(current_status)
    
    # Găsește ultima înregistrare video creată din directorul VIDEO_OUTPUT_DIR
    video_files = [os.path.join(VIDEO_OUTPUT_DIR, f) for f in os.listdir(VIDEO_OUTPUT_DIR) 
                  if f.endswith('.mp4')]
    
    if not video_files:
        error_msg = "Nu s-a găsit niciun fișier video generat."
        print(error_msg)
        return None, error_msg
    
    video_path = max(video_files, key=os.path.getctime)
    success_msg = f"Video generat cu succes la: {video_path}"
    print(success_msg)
    
    return video_path, success_msg

# Interfața Gradio
def create_interface():
    with gr.Blocks(title="SkyReels-V2 Demo") as demo:
        gr.Markdown("# SkyReels-V2 Demo Generator")
        gr.Markdown("Generați videoclipuri din imagini sau text folosind modelele SkyReels-V2")
        
        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Descrieți imaginea dorită", lines=3)
                    t2i_steps = gr.Slider(minimum=20, maximum=100, value=50, step=1, label="Pași de inferență")
                    t2i_guidance = gr.Slider(minimum=1, maximum=15, value=7.5, step=0.1, label="Guidance Scale")
                    t2i_seed = gr.Number(label="Seed (opțional)", precision=0)
                    t2i_button = gr.Button("Generează Imaginea")
                
                with gr.Column():
                    t2i_output = gr.Image(label="Imagine Generată")
                    t2i_path = gr.Textbox(label="Calea către imaginea generată")
        
        with gr.Tab("Image to Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    i2v_image = gr.Image(label="Încărcați o imagine sau generați una din tab-ul Text to Image", type="pil")
                    i2v_prompt = gr.Textbox(label="Descrieți video-ul dorit", lines=3)
                    
                    with gr.Row():
                        with gr.Column():
                            i2v_resolution = gr.Dropdown(choices=["540P", "720P"], value="720P", label="Rezoluție")
                            i2v_frames = gr.Slider(minimum=97, maximum=257, value=121, step=1, label="Număr de cadre")
                            i2v_guidance = gr.Slider(minimum=1, maximum=10, value=5.0, step=0.1, label="Guidance Scale")
                            i2v_shift = gr.Slider(minimum=1, maximum=10, value=3.0, step=0.1, label="Shift")
                            i2v_fps = gr.Slider(minimum=15, maximum=60, value=24, step=1, label="FPS")
                            i2v_seed = gr.Number(label="Seed (opțional)", precision=0)
                        
                        with gr.Column():
                            i2v_offload = gr.Checkbox(value=True, label="Offload")
                            i2v_teacache = gr.Checkbox(value=True, label="Folosește Teacache")
                            i2v_use_ret_steps = gr.Checkbox(value=True, label="Folosește Retention Steps")
                            i2v_teacache_thresh = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Teacache Threshold")
                    
                    i2v_button = gr.Button("Generează Video", variant="primary")
                
                with gr.Column(scale=2):
                    i2v_output = gr.Video(label="Video Generat")
                    i2v_message = gr.Textbox(label="Mesaj Status")
        
        # Conectați butoanele la funcții
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
    # Creează și lansează interfața
    interface = create_interface()
    interface.queue()  # Activează sistemul de coadă Gradio
    interface.launch(share=True, debug=True)
