import time
import cv2
import os
from datetime import datetime
import face_recognition_models
import face_recognition
import numpy as np



def list_cameras(max_test=3):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def trigger_alarm(times=3, interval=0.3):
    for _ in range(times):
        os.system("play -n synth 0.1 sine 5000")
        time.sleep(interval)

def register_log(filename="motion_detection_log.txt", content="Motion detected"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"[{now} ] {content}\n")


def record_video(cap, video_name, duration_seconds=5, fps=20.0):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    frames_recorded = 0
    max_frames = int(fps * duration_seconds)

    while frames_recorded < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_recorded += 1

    out.release()
    



def detect_faces(frame, known_face_encodings, known_face_names):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            register_log(filename='face_detection_log.txt', content=f'{name} detected!')
        else:
            
            timestamp = int(time.time())
            filename = f"face_{timestamp}.jpg"
            filepath = os.path.join('detected_faces', filename)
            
            cv2.imwrite(filepath, frame)

            register_log(filename='face_detection_log.txt', content=f'{filename} detected!')
            known_face_encodings.append(face_encoding)
            known_face_names.append(f"person_{timestamp}")
            


def load_encondings(known_face_encodings, known_face_names):
    dir = "detected_faces"
    for filename in os.listdir(dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image, model="large")

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename)

    return known_face_encodings, known_face_names

