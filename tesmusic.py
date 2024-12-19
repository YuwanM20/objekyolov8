import requests
import time
import os
from pygame import mixer

# Firebase configuration
firebase_url = "https://tesdatayolov8-default-rtdb.asia-southeast1.firebasedatabase.app/"
firebase_api_key = "AIzaSyAbKnqoV2D0GdmmmFhfl0AZeuui9NoZ9uM"

# Music mapping
music_files = {
    1: "sound/kursi.mp3",
    2: "sound/pintu.mp3",
    3: "sound/orang.mp3",
    4: "sound/meja.mp3",
    5: "sound/kosong.mp3"
}

# Delay for object 5 (in seconds)
delay_for_object_5 = 5

# Initialize pygame mixer
mixer.init()

def get_firebase_data():
    try:
        response = requests.get(f"{firebase_url}/detections.json?auth={firebase_api_key}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def play_music(detected_object):
    music_file = music_files.get(detected_object)
    if music_file and os.path.exists(music_file):
        print(f"Playing music: {music_file}")
        mixer.music.load(music_file)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)  # Wait for the music to finish
    else:
        print(f"Music file not found or invalid for detected object: {detected_object}")

if __name__ == "__main__":
    print("Starting Firebase detection and music player...")
    last_detected_object = None
    last_play_time_5 = 0

    while True:
        data = get_firebase_data()
        if data and "detected_object" in data:
            detected_object = data["detected_object"]
            current_time = time.time()

            if detected_object == 5:
                if current_time - last_play_time_5 >= delay_for_object_5:
                    play_music(detected_object)
                    last_play_time_5 = current_time
            elif detected_object != last_detected_object:
                play_music(detected_object)
                last_detected_object = detected_object
        else:
            print("No valid detected_object found in data.")

        time.sleep(1)
