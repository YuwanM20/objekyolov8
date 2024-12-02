import cv2
from ultralytics import YOLO
from gtts import gTTS
import os
import pygame

# Inisialisasi pygame mixer untuk memutar audio
pygame.mixer.init()

def speak_with_gTTS(text):
    """Menggunakan Google Text-to-Speech untuk menghasilkan suara."""
    tts = gTTS(text=text, lang='id')  # 'id' untuk Bahasa Indonesia
    audio_file = "temp_audio.mp3"  # File sementara untuk audio
    tts.save(audio_file)
    
    # Putar audio dengan pygame
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue  # Tunggu hingga audio selesai diputar

    # Hapus file audio sementara
    if os.path.exists(audio_file):
        os.remove(audio_file)

# Path model YOLOv8
model_path = "model.pt"
model = YOLO(model_path)

# Inisialisasi kamera (gunakan 0 untuk kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

print("Tekan 'q' untuk keluar.")

# Daftar kelas yang akan dideteksi
target_classes = ["person", "chair", "table", "door"]  # Ganti sesuai kelas pada model Anda

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Deteksi objek menggunakan YOLOv8
    results = model.predict(frame, conf=0.5)  # Ubah conf jika diperlukan
    annotated_frame = results[0].plot()  # Annotasi frame dengan hasil deteksi

    # Inisialisasi variabel untuk melacak objek yang terdeteksi
    detected_objects = []

    # Iterasi melalui hasil deteksi
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Indeks kelas
            label = result.names[cls]  # Nama kelas
            
            if label in target_classes:
                detected_objects.append(label)  # Tambahkan objek yang terdeteksi ke daftar
                
                # Output suara sesuai objek yang terdeteksi
                if label == "person":
                    speak_with_gTTS("Terdapat orang di depan.")
                elif label == "chair":
                    speak_with_gTTS("Terdapat kursi di depan.")
                elif label == "table":
                    speak_with_gTTS("Terdapat meja di depan.")
                elif label == "door":
                    speak_with_gTTS("Terdapat pintu di depan.")

    # Jika tidak ada objek terdeteksi
    if not detected_objects:
        speak_with_gTTS("Di depan kosong, Anda bisa jalan lurus.")

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
