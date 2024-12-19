import cv2
import threading
import queue
import time
from ultralytics import YOLO
import gradio as gr
import requests

# Model YOLOv8
model_path = "model_n.pt"
model = YOLO(model_path).to("cpu")

# Frame queue for threading
frame_queue = queue.Queue(maxsize=2)  # Tingkatkan ukuran jika diperlukan

def send_to_firebase(data, firebase_url, firebase_api_key):
    try:
        url = f"{firebase_url}/detections.json"
        response = requests.put(url, json=data, params={"auth": firebase_api_key})
        if response.status_code == 200:
            print("Data berhasil dikirim ke Firebase:", data)
        else:
            print("Gagal mengirim data ke Firebase:", response.content)
    except Exception as e:
        print("Error mengirim data ke Firebase:", e)

def capture_frames(ip_webcam_url):
    cap = cv2.VideoCapture(ip_webcam_url)
    if not cap.isOpened():
        raise Exception("Tidak dapat terhubung ke IP Webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Kurangi resolusi untuk kinerja lebih baik
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

def process_detection(firebase_url, firebase_api_key):
    if frame_queue.empty():
        return None, None

    frame = frame_queue.get()

    # Proses hanya setiap beberapa frame (contoh: setiap 5 frame)
    if int(time.time() * 10) % 5 != 0:
        return frame, frame  # Skip deteksi untuk beberapa frame

    results = model.predict(frame, conf=0.5)
    detections = results[0].boxes

    detection_counts = {}
    for box in detections:
        class_id = int(box.cls[0])
        detection_counts[class_id] = detection_counts.get(class_id, 0) + 1

    data = {"detected_object": detection_counts}
    send_to_firebase(data, firebase_url, firebase_api_key)

    annotated_frame = results[0].plot()
    return frame, annotated_frame

def start_detection(ip_webcam_url, firebase_url, firebase_api_key):
    capture_thread = threading.Thread(target=capture_frames, args=(ip_webcam_url,))
    capture_thread.daemon = True
    capture_thread.start()

    while True:
        frame, annotated_frame = process_detection(firebase_url, firebase_api_key)
        if frame is not None and annotated_frame is not None:
            yield frame[:, :, ::-1], annotated_frame[:, :, ::-1]

def gradio_interface():
    def detection_runner(ip_webcam_url, firebase_url, firebase_api_key):
        generator = start_detection(ip_webcam_url, firebase_url, firebase_api_key)
        for raw_frame, detected_frame in generator:
            yield raw_frame, detected_frame

    with gr.Blocks() as demo:
        with gr.Row():
            ip_webcam_url = gr.Textbox(label="IP Webcam URL", value="http://192.168.151.14:8080/video")
            firebase_url = gr.Textbox(label="Firebase URL", value="https://tesdatayolov8-default-rtdb.asia-southeast1.firebasedatabase.app/")
            firebase_api_key = gr.Textbox(label="Firebase API Key", value="AIzaSyAbKnqoV2D0GdmmmFhfl0AZeuui9NoZ9uM", type="password")
        with gr.Row():
            raw_frame = gr.Image(label="Camera Feed")
            detected_frame = gr.Image(label="Detected Objects")
        start_button = gr.Button("Start Detection")

        start_button.click(
            detection_runner,
            inputs=[ip_webcam_url, firebase_url, firebase_api_key],
            outputs=[raw_frame, detected_frame]
        )

    demo.launch()

gradio_interface()
