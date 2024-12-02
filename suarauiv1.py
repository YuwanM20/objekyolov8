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
model_path = "model.pt"  # Pastikan Anda memiliki model nano untuk percepatan
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
    "person": "sound/orang.mp3",  # Ganti dengan path file audio untuk "person"
    "chair": "sound/kursi.mp3",    # Ganti dengan path file audio untuk "chair"
    "table": "sound/meja.mp3",    # Ganti dengan path file audio untuk "table"
    "door": "sound/pintu.mp3"       # Ganti dengan path file audio untuk "door"
}

# Suara untuk area kosong
empty_sound = "sound/kosong.mp3"  # Ganti dengan path file audio untuk "Di depan kosong"

# Variabel waktu terakhir untuk cooldown
last_detection_time = time.time()
cooldown = 1.5  # Waktu cooldown antara suara (dalam detik)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Resize frame untuk mempercepat inferensi (tanpa mengubah tampilan kamera)
    small_frame = cv2.resize(frame, (640, 480))  # Resolusi lebih kecil untuk YOLOv8

    # Deteksi objek menggunakan YOLOv8
    results = model.predict(small_frame, conf=0.5, imgsz=640)  # Optimalkan ukuran gambar
    annotated_frame = results[0].plot()  # Annotasi frame dengan hasil deteksi

    # Inisialisasi variabel untuk melacak objek yang terdeteksi
    detected_objects = []

    # Iterasi melalui hasil deteksi
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Indeks kelas
            label = result.names[cls]  # Nama kelas
            
            if label in sound_mapping and label not in detected_objects:
                detected_objects.append(label)  # Tambahkan objek yang terdeteksi ke daftar
                
                # Output suara jika waktu cooldown terpenuhi
                current_time = time.time()
                if current_time - last_detection_time > cooldown:
                    play_sound(sound_mapping[label])
                    last_detection_time = current_time

    # Jika tidak ada objek terdeteksi
    if not detected_objects:
        current_time = time.time()
        if current_time - last_detection_time > cooldown:
            play_sound(empty_sound)  # Memutar suara untuk area kosong
            last_detection_time = current_time

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
