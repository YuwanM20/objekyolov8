import cv2
import threading
from ultralytics import YOLO

# Ganti dengan path model YOLOv8 yang telah Anda latih
model_path = "model_n.pt"
model = YOLO(model_path)

# Ganti dengan URL IP Webcam Anda
ip_webcam_url = "http://192.168.151.14:8080/video"  # Sesuaikan dengan IP dan port IP Webcam
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("Error: Tidak dapat terhubung ke IP Webcam.")
    exit()

# Inisialisasi variabel untuk multithreading
frame_lock = threading.Lock()
latest_frame = None
stop_thread = False

def read_frame():
    """Fungsi untuk membaca frame dari kamera secara terus-menerus."""
    global latest_frame, stop_thread
    while not stop_thread:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame

def process_detection():
    """Fungsi untuk melakukan deteksi objek pada frame terbaru."""
    global latest_frame, stop_thread
    while not stop_thread:
        if latest_frame is not None:
            with frame_lock:
                frame = latest_frame.copy()
            # Deteksi objek menggunakan YOLOv8
            results = model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Tampilkan hasil deteksi
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_thread = True
                break

# Jalankan thread untuk membaca frame
read_thread = threading.Thread(target=read_frame)
read_thread.start()

# Jalankan deteksi di thread utama
process_detection()

# Berhenti dan bersihkan
stop_thread = True
read_thread.join()
cap.release()
cv2.destroyAllWindows()
