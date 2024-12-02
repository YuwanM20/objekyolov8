import cv2
from ultralytics import YOLO
import pyttsx3

# Inisialisasi Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Kecepatan bicara
tts_engine.setProperty('volume', 1.0)  # Volume maksimum

def speak(text):
    """Fungsi untuk mengucapkan teks"""
    tts_engine.say(text)
    tts_engine.runAndWait()

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
                    speak("Terdapat orang di depan.")
                elif label == "chair":
                    speak("Terdapat kursi di depan.")
                elif label == "table":
                    speak("Terdapat meja di depan.")
                elif label == "door":
                    speak("Terdapat pintu di depan.")

    # Jika tidak ada objek terdeteksi
    if not detected_objects:
        speak("Di depan kosong, Anda bisa jalan lurus.")

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
