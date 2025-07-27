import os
import time
import cv2
from utils import list_cameras, load_encondings, record_video, trigger_alarm, register_log, detect_faces
from datetime import datetime
import argparse



def main():
    os.makedirs("detected_images", exist_ok=True)
    os.makedirs("detected_videos", exist_ok=True)
    os.makedirs("detected_faces", exist_ok=True)

    parser = argparse.ArgumentParser(description="Motion detection script")
    parser.add_argument("--modes", nargs="+", choices=["video-only", "image-only", "face-only"],)
    parser.add_argument("--silent", action="store_true")


    args = parser.parse_args()

    
    
    cameras = list_cameras()

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

    known_face_encodings = [] 
    known_face_names = [] 
    load_encondings(known_face_encodings, known_face_names)
    

    
    
    
    print("🎥 Monitoring... Press CTRL + C to exit.")
    while cap.isOpened():
        ret, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if not ret or not ret2:
            print("Error capturing initial frames.")
            exit(1)
        


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
            print(f"\033[91mMotion detected at {now.strftime('%H:%M:%S')}!\033[0m")

            
            if args.modes is None or "face-only" in args.modes:
                detect_faces(frame2, known_face_encodings, known_face_names)


            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            

            if args.modes is None or "image-only" in args.modes:
                image_name = f"detected_images/capture_{timestamp_str}.jpg"
                cv2.imwrite(image_name, frame1)

            
            if not args.silent:
                trigger_alarm()
            register_log()
            
            if args.modes is None or "video-only" in args.modes:
                video_name = f"detected_videos/detection_{timestamp_str}.avi"
                record_video(cap, video_name)
    
        
            time.sleep(3) 

            frame1 = frame2
            ret, frame2 = cap.read()

        if not ret:
            break

        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()