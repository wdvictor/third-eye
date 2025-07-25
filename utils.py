import time
import cv2
import os
from datetime import datetime

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

def register_log(text, filename="motion_detection_log.txt"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"[{now}] {text}\n")
