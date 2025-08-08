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
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500)
    utils.preview_camera(cap)
    args = utils.init_configuration()


    utils.load_encondings()
    
    print("🎥 Monitoring... Press CTRL + C to exit.")
    try:
        while cap.isOpened():
            
            frame = utils.get_frame(cap)
           
            
            mask = bgSubtractor.apply(frame, learningRate=0.005)
            motion_detected = utils.detect_motion(mask)

            if motion_detected:
                now = datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

                pretty_print(MessageType.ALERT.name, "Motion detected")

                if not args.silent:
                    utils.trigger_alarm()
                    
                
                if args.modes is None or "image" in args.modes:
                    image_name = f"{utils.detected_images_dir}/capture_{timestamp_str}.jpg"
                    cv2.imwrite(image_name, frame)
                
                if args.modes is not None and  "video" in args.modes:
                    utils.record_video(cap)

                if args.modes is None or "face-detection" in args.modes:
                    t = threading.Thread(target=utils.detect_and_recognize_faces, kwargs={'frame': frame})
                    t.start()

            if motion_detected:
                time.sleep(5)  
                
                
    finally:
        cap.release()


if __name__ == "__main__":
    main()