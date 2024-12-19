import cv2
import threading
import queue
import time
from ultralytics import YOLO
import requests  # Untuk mengirim data ke Firebase

# Ganti dengan path model YOLOv8 yang telah Anda latih
model_path = "model_n.pt"
model = YOLO(model_path).to("cpu")  # Gunakan GPU jika tersedia

# Ganti dengan URL IP Webcam Anda
ip_webcam_url = "http://192.168.151.14:8080//video"  
cap = cv2.VideoCapture(ip_webcam_url)

# Set resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Tidak dapat terhubung ke IP Webcam.")
    exit()

# Firebase configuration
firebase_url = "https://tesdatayolov8-default-rtdb.asia-southeast1.firebasedatabase.app/"
firebase_api_key = "AIzaSyAbKnqoV2D0GdmmmFhfl0AZeuui9NoZ9uM"

# Daftar kelas objek
names = ['chair', 'door', 'person', 'table', 'other']

# Frame queue untuk threading
frame_queue = queue.Queue(maxsize=1)

# Interval pengiriman data ke Firebase
firebase_update_interval = 2  # Kirim data ke Firebase setiap 2 detik
last_firebase_time = 0


def send_to_firebase(data):
    """Kirim data deteksi ke Firebase."""
    global last_firebase_time
    current_time = time.time()
    if current_time - last_firebase_time < firebase_update_interval:
        return  # Batasi pengiriman data berdasarkan interval

    try:
        url = f"{firebase_url}/detections.json"  # Kirim ke satu field `detected_object`
        response = requests.put(url, json=data, params={"auth": firebase_api_key})
        if response.status_code == 200:
            print("Data berhasil dikirim ke Firebase:", data)
        else:
            print("Gagal mengirim data ke Firebase:", response.content)
    except Exception as e:
        print("Error mengirim data ke Firebase:", e)

    last_firebase_time = current_time


def capture_frames():
    """Fungsi untuk menangkap frame dari kamera dan menaruhnya di buffer."""
    global frame_queue
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame dari kamera.")
            break

        if frame_queue.full():
            frame_queue.get()  # Hapus frame lama untuk buffer baru
        frame_queue.put(frame)


def process_detection():
    """Fungsi untuk melakukan deteksi objek secara terpisah dari pengambilan frame."""
    global frame_queue
    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        # Deteksi objek menggunakan YOLOv8
        results = model.predict(frame, conf=0.5)
        detections = results[0].boxes  # Mendapatkan bounding box dari hasil deteksi
        detection_counts = [0] * len(names)  # Array untuk menghitung jumlah tiap kelas

        for box in detections:
            class_id = int(box.cls[0])  # Mendapatkan ID kelas objek
            if class_id < len(names):
                detection_counts[class_id] += 1  # Hitung setiap objek terdeteksi

        # Tentukan angka yang dikirim ke Firebase berdasarkan deteksi objek
        if sum(detection_counts) == 0:
            detected_object = 4  # Jika tidak ada objek terdeteksi, kirim '4' untuk 'other'
        else:
            # Kirim angka yang sesuai dengan urutan yang diinginkan
            # 5: door, 3: chair, 1: person, 2: table
            detected_object = {
                1: 3,  # 'person' -> 3
                2: 5,  # 'table' -> 2
                3: 1,  # 'chair' -> 1
                4: 2,  # 'door' -> 5
                5: 4   # 'other' -> 4
            }.get(max(range(len(detection_counts)), key=lambda i: detection_counts[i]) + 1, 4)

        # Kirim data ke Firebase
        data = {"detected_object": detected_object}
        send_to_firebase(data)

        # Annotasi dan tampilkan hasil deteksi
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Jalankan Thread untuk menangkap frame
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Jalankan deteksi objek
process_detection()

# Berhenti dan bersihkan
cap.release()
cv2.destroyAllWindows()
