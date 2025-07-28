import json
import os
import time
import cv2
from utils  import Utils
from datetime import datetime
import argparse
import threading



def main():
    utils = Utils()
    os.makedirs(utils.detected_images_dir, exist_ok=True)
    os.makedirs(utils.detected_videos_dir, exist_ok=True)
    os.makedirs(utils.detected_faces_dir, exist_ok=True)
    

    if not os.path.exists(utils.json_cache_file):
        cache_data = {"number_faces_detected": 0}
        with open(utils.json_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4, ensure_ascii=False)
            
       
    parser = argparse.ArgumentParser(description="Motion detection script")
    parser.add_argument("--modes", nargs="+", choices=["video-only", "image-only", "face-only"],)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    
    
    cameras = utils.list_cameras()

    if len(cameras) == 0:
        print("No cameras detected. Exiting.")
        exit(1)
    elif len(cameras) == 1:
        selected_camera = cameras[0]
        print(f"Only one camera detected. Using camera index {selected_camera}.")
    else:
        print("Multiple cameras detected:")
        for i, cam_idx in enumerate(cameras):
            print(f" [{i}] Camera index {cam_idx}")
        choice = input(f"Choose camera [0-{len(cameras)-1}]: ")
        try:
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]
            else:
                print("Invalid choice. Using first camera by default.")
                selected_camera = cameras[0]
        except:
            print("Invalid input. Using first camera by default.")
            selected_camera = cameras[0]

    cap = cv2.VideoCapture(selected_camera)
    
    time.sleep(2)

    utils.preview_camera(cap)

    known_face_encodings = [] 
    known_face_names = [] 
    utils.load_encondings(known_face_encodings, known_face_names)
    
    
    print("🎥 Monitoring... Press CTRL + C to exit.")
    try:
        while cap.isOpened():


            _, frame1 = cap.read()
            _, frame2 = cap.read()
            
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            
            for contour in contours:
                
                if cv2.contourArea(contour) < 1000:
                    continue

                motion_detected = True
                break

            if motion_detected:
                now = datetime.now()

                print(f"\033[91mMotion detected!\033[0m")

                if not args.silent:
                    utils.trigger_alarm()
                    
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                utils.register_log()
                
                if args.modes is None or "image-only" in args.modes:
                    image_name = f"{utils.detected_images_dir}/capture_{timestamp_str}.jpg"
                    cv2.imwrite(image_name, frame1)
                
                if args.modes is None or "video-only" in args.modes:
                    utils.record_video(cap)

                if args.modes is None or "face-only" in args.modes:
                    t = threading.Thread(target=utils.detect_faces, args=(frame2, known_face_encodings, known_face_names,))
                    t.start()

                
                time.sleep(3)
                
    finally:
        cap.release()


if __name__ == "__main__":
    main()