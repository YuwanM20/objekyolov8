import cv2
from ultralytics import YOLO

# Ganti dengan path model YOLOv8 yang telah Anda latih
model_path = "model.pt"
model = YOLO(model_path)

# Inisialisasi kamera (gunakan 0 untuk kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Deteksi objek menggunakan YOLOv8
    results = model.predict(frame, conf=0.5)  # Ubah conf jika diperlukan
    annotated_frame = results[0].plot()  # Annotasi frame dengan hasil deteksi

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
