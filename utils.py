import threading
import time
import cv2
import os
from datetime import datetime
import face_recognition
from config import detected_faces_dir, face_detection_log,  motion_log_detection_log, detected_images_dir, detected_videos_dir, json_cache_file
import json



def save_cache(json_data):
    with open(json_cache_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    
def load_json_cache():
    with open(json_cache_file, "r", encoding="utf-8") as f:
        return json.load(f)
    


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
        os.system("play -n synth 0.1 sine 5000 > /dev/null 2>&1")
        time.sleep(interval)


def register_log(filename=motion_log_detection_log, content="Motion detected"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"[{now} ] {content}\n")


def record_video(cap, duration_seconds=5, fps=20):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    video_name = f"{detected_videos_dir}/detection_{timestamp_str}.avi"
                
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)
    
    
    frames_recorded = 0
    max_frames = int(fps * duration_seconds * 2)
    

    while frames_recorded < max_frames:
        _, frame = cap.read()
        out.write(frame)
        frames_recorded += 1
        
    out.release()
    
    

    

def detect_faces(frame, known_face_encodings, known_face_names):
    
    cache = load_json_cache()
    person_id = int(cache["number_faces_detected"])

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn", number_of_times_to_upsample=2)

    if face_locations == []:
        return
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=50)


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            register_log(filename=face_detection_log, content=f'{name} detected!')
            return
        else:
            person_id += 1
            cache["number_faces_detected"] = person_id
            save_cache(cache)

            filename = f"person_{person_id}.jpg"
            filepath = os.path.join(detected_faces_dir, filename)
            
            cv2.imwrite(filepath, frame)

            register_log(filename=face_detection_log, content=f'person_{person_id} detected!')
            known_face_encodings.append(face_encoding)
            known_face_names.append(f"person_{person_id}")
            return
            

def load_encondings(known_face_encodings, known_face_names):
    for filename in os.listdir(detected_faces_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(detected_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image, model="large")

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename.replace(".jpg", "").replace(".png", "").replace(".jpeg", "") )

    return known_face_encodings, known_face_names


def preview_camera(cap):
    while True:
        _, frame = cap.read()
        
        cv2.imshow("Webcam Preview - Press 'q' to continue", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



