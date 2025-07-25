import os
import time
import cv2
from utils import list_cameras, trigger_alarm, register_log
from datetime import datetime
import argparse



def main():
    os.makedirs("detected_images", exist_ok=True)
    os.makedirs("detected_videos", exist_ok=True)

    parser = argparse.ArgumentParser(description="Motion detection script")
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        help='''The operation mode. Possible values are: [detection] (There will be other values soon).'''
        '''If the parameter is not provided, all modes will be executed'''
    )
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

    print(f"Using camera index: {selected_camera}")


    cap = cv2.VideoCapture(selected_camera)
    
    time.sleep(2)

    

    
    
    if(args.mode is None or args.mode == "detection"):
        print("🎥 Monitoring... Press ESC to exit.")
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
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"detected_images/capture_{timestamp_str}.jpg"
                video_name = f"detected_videos/detection_{timestamp_str}.avi"

                cv2.imwrite(image_name, frame1)

                print(f"Motion detected at {now.strftime('%H:%M:%S')}!")
                trigger_alarm()
                register_log(f"Motion detected - Image: {image_name}, Video: {video_name}")

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 20.0
                frame_size = (int(cap.get(3)), int(cap.get(4)))
                out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

                frames_recorded = 0
                max_frames = int(fps * 5)  

                while frames_recorded < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frames_recorded += 1
                    
                
                   
                if cv2.waitKey(1) == 27:  
                    break

                out.release()
                

                time.sleep(3) 

                frame1 = frame2
                ret, frame2 = cap.read()

            if not ret:
                break

            if cv2.waitKey(1) == 27:  
                print("🛑 Monitoring stopped.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()