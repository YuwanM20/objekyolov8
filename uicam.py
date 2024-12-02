import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Ganti dengan path model YOLOv8 yang telah Anda latih
model_path = "model.pt"
model = YOLO(model_path)

def detect_objects(image):
    # Konversi gambar dari format PIL ke array numpy (OpenCV format)
    image = np.array(image)
    
    # Deteksi objek menggunakan YOLOv8
    results = model.predict(image, conf=0.5)  # Confidence threshold dapat disesuaikan
    annotated_image = results[0].plot()  # Annotasi hasil deteksi
    
    # Kembalikan hasil deteksi sebagai gambar
    return annotated_image

# UI dengan Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Deteksi Objek dengan YOLOv8")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Gambar", type="pil")
            detect_button = gr.Button("Deteksi")
        
        with gr.Column():
            image_output = gr.Image(label="Hasil Deteksi")
    
    detect_button.click(detect_objects, inputs=image_input, outputs=image_output)

# Jalankan aplikasi
demo.launch()
