import json
import os
import time
import cv2
from pretty_print import MessageType, pretty_print
from utils  import Utils
from datetime import datetime
import threading



def main():

    utils = Utils()
    cap = utils.select_camera()
    utils.preview_and_subtract_background(cap)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    args, face_detection, face_recognition = utils.init_configuration(frame_size)

    
   

    known_face_encodings = [] 
    known_face_names = [] 
    utils.load_encondings(known_face_encodings, known_face_names)
    
    print("🎥 Monitoring... Press CTRL + C to exit.")
    try:
        while cap.isOpened():

            _, frame1 = utils.get_frame(cap)
            _, frame2 = utils.get_frame(cap)
            
            motion_detected = utils.detect_motion()

            if motion_detected:
                now = datetime.now()

                pretty_print(MessageType.ALERT.name, "Motion detected")

                if not args.silent:
                    utils.trigger_alarm()
                    
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                utils.register_log()
                
                if args.modes is None or "image" in args.modes:
                    image_name = f"{utils.detected_images_dir}/capture_{timestamp_str}.jpg"
                    cv2.imwrite(image_name, frame2)
                
                if "video" in args.modes:
                    utils.record_video(cap)

                if args.modes is None or "face-detection" in args.modes:
                    t = threading.Thread(target=utils.detect_faces, args=(frame2, known_face_encodings, known_face_names,))
                    t.start()

                
                time.sleep(3)
                
    finally:
        cap.release()


if __name__ == "__main__":
    main()