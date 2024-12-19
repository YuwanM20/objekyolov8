import cv2
from ultralytics import YOLO
import pygame
import time

# Inisialisasi pygame mixer untuk memutar audio
pygame.mixer.init()

def play_sound(file_name):
    """Memutar file audio berdasarkan nama file."""
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue  # Tunggu hingga audio selesai diputar

# Path model YOLOv8 (gunakan model ringan jika tersedia)
model_path = "model_n.pt" 
model = YOLO(model_path)

# Inisialisasi kamera (gunakan 0 untuk kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

# Atur resolusi kamera ke nilai yang lebih kecil untuk performa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar.")

# Daftar kelas yang akan dideteksi dan file audio terkait
sound_mapping = {
    "person": "sound/orang.mp3",
    "chair": "sound/kursi.mp3",
    "table": "sound/meja.mp3",
    "door": "sound/pintu.mp3"
}

# Suara untuk area kosong
empty_sound = "sound/kosong.mp3"

# Set objek yang sudah diputar suaranya
played_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Resize frame untuk mempercepat inferensi
    small_frame = cv2.resize(frame, (640, 480))

    # Deteksi objek menggunakan YOLOv8
    results = model.predict(small_frame, conf=0.5, imgsz=640)
    annotated_frame = results[0].plot()

    # Objek yang terdeteksi pada frame ini
    detected_objects = set()

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Indeks kelas
            label = result.names[cls]  # Nama kelas
            
            if label in sound_mapping:
                detected_objects.add(label)  # Tambahkan ke objek yang terdeteksi

    # Periksa objek baru yang belum diputar suaranya
    new_objects = detected_objects - played_objects

    if new_objects:
        # Mainkan suara untuk objek baru
        for obj in new_objects:
            play_sound(sound_mapping[obj])
            played_objects.add(obj)
    elif not detected_objects:
        # Jika tidak ada objek, mainkan suara kosong
        if played_objects:
            play_sound(empty_sound)
            played_objects.clear()

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
