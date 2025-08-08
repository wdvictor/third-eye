import argparse
import pickle
import time
import cv2
import os
from datetime import datetime
import face_recognition
import json
import numpy as np
from pretty_print import MessageType, pretty_print


class Utils:
    
    def __init__(self):
        self.detected_faces_dir = "detected_faces"
        self.database_dir = "face_database"
        self.detected_images_dir = "detected_images"
        self.detected_videos_dir = "detected_videos"
        

        self.motion_log_detection_log = "motion_detection_log.txt"
        self.log_file = "face_detection_log.json"

        os.makedirs(self.detected_images_dir, exist_ok=True)
        os.makedirs(self.detected_videos_dir, exist_ok=True)
        os.makedirs(self.detected_faces_dir, exist_ok=True)
        os.makedirs(self.database_dir, exist_ok=True)

        self.known_faces = []
        self.known_names = []
        self.face_counter = 0
        
        
        self.load_known_faces()
        
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        pretty_print(MessageType.SUCCESS.name, f" System initialized!")
        pretty_print(MessageType.SUCCESS.name, f"📁 Faces saved in: {self.detected_faces_dir}")
        pretty_print(MessageType.SUCCESS.name, f"🗃️  Database in: {self.database_dir}")
        pretty_print(MessageType.SUCCESS.name, f"📋 Log in: {self.log_file}")



    def init_configuration(self):
        
        if not os.path.exists(self.log_file):
            cache_data = {"number_faces_detected": 0}
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=4, ensure_ascii=False)
                
        
        parser = argparse.ArgumentParser(description="Motion detection script")
        parser.add_argument("--modes", nargs="+", choices=["video", "image", "face-detection"],)
        parser.add_argument("--silent", action="store_true")
        args = parser.parse_args()

        self.known_faces = []
        self.known_names = []
        self.face_counter = 0
        
        self.load_known_faces()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        pretty_print(MessageType.SUCCESS.name,f"✅ System initialized!")
        pretty_print(MessageType.SUCCESS.name,f"📁 Faces saved in: {self.detected_faces_dir}")
        pretty_print(MessageType.SUCCESS.name,f"🗃️  Database in: {self.database_dir}")
        pretty_print(MessageType.SUCCESS.name,f"📋 Log in: {self.log_file}")


        return args

        
    def load_known_faces(self):
        """Loads known faces from the database"""
        database_file = os.path.join(self.database_dir, "faces_database.pkl")
        
        if os.path.exists(database_file):
            try:
                with open(database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                    self.face_counter = data.get('counter', 0)
                pretty_print(MessageType.SUCCESS.name, f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                pretty_print(MessageType.FAILURE.name,f"Error loading database: {e}")
        else:
            pretty_print(MessageType.ALERT.name,"Creating new face database")    

    def save_known_faces(self):
        """Saves known faces to the database"""
        database_file = os.path.join(self.database_dir, "faces_database.pkl")
        
        try:
            data = {
                'faces': self.known_faces,
                'names': self.known_names,
                'counter': self.face_counter
            }
            with open(database_file, 'wb') as f:
                pickle.dump(data, f)
            pretty_print(MessageType.SUCCESS.name, "Database saved successfully")
        except Exception as e:
            pretty_print(MessageType.FAILURE.name,"Error saving database: {e}")


    def generate_face_name(self):
        """Generates a unique name for a new face"""

        self.face_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"Person_{self.face_counter:03d}_{timestamp}"
    

    def log_detection(self, name, is_new_face=False, confidence=0.0):
        """Logs detection in the log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "is_new_face": is_new_face,
            "confidence": confidence
        }
        
        log_data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = []
        
        log_data.append(log_entry)
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            pretty_print(MessageType.FAILURE.name, f"Error saving log: {e}")
    

    def detect_and_recognize_faces(self, frame):
        """
        Detects and recognizes faces in the frame
        
        Returns:
            list: List of dictionaries with information about detected faces
        """
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        detected_faces = []
        
        for i, (face_encoding, face_locations) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_locations
            
            
            if len(self.known_faces) > 0:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    
                    if matches[best_match_index] and confidence > 0.4:
                        
                        name = self.known_names[best_match_index]
                        is_new = False
                        pretty_print(MessageType.ALERT.name, f"Recognized face: {name} (confidence: {confidence:.2f})")
                    else:
                        
                        name = self.generate_face_name()
                        is_new = True
                        self.known_faces.append(face_encoding)
                        self.known_names.append(name)
                        confidence = 1.0
                        pretty_print(MessageType.ALERT.name, f"New face detected: {name}")
                else:
                    
                    name = self.generate_face_name()
                    is_new = True
                    self.known_faces.append(face_encoding)
                    self.known_names.append(name)
                    confidence = 1.0
                    pretty_print(MessageType.ALERT.name, f"First face detected: {name}")
            else:
                # First face in the system
                name = self.generate_face_name()
                is_new = True
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                confidence = 1.0
                pretty_print(MessageType.ALERT.name, f"First face detected: {name}")
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(self.detected_faces_dir, filename)
            cv2.imwrite(filepath, frame)
            pretty_print(MessageType.SUCCESS.name, f" Face saved: {filename}")
            
            
            self.log_detection(name, is_new, confidence)
            
            
            detected_faces.append({
                'name': name,
                'location': (left, top, right, bottom),
                'confidence': confidence,
                'is_new': is_new
            })
        
       
        if any(face['is_new'] for face in detected_faces):
            self.save_known_faces()
        
        return detected_faces


    def detect_motion(self, frame, min_area=800):
        """Detects motion based on the mask"""

        bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
        mask = bgSubtractor.apply(frame, learningRate=0.1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                valid_contours.append(contour)
        
        return len(valid_contours) > 0, valid_contours


    def list_cameras(self,max_test=3):
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available


    def select_camera(self):
        cameras = self.list_cameras()
        if len(cameras) == 0:
            pretty_print(MessageType.SUCCESS.name,"No cameras detected. Exiting.")
            exit(1)
        elif len(cameras) == 1:
            selected_camera = cameras[0]
            pretty_print(MessageType.SUCCESS.name,f"Only one camera detected. Using camera index {selected_camera}.")
        else:
            pretty_print(MessageType.SUCCESS.name,"Multiple cameras detected:")
            for i, cam_idx in enumerate(cameras):
                pretty_print(MessageType.SUCCESS.name,f" [{i}] Camera index {cam_idx}")
            choice = input(f"Choose camera [0-{len(cameras)-1}]: ")

            try:
                choice_idx = int(choice)
                if 0 <= choice_idx < len(cameras):
                    selected_camera = cameras[choice_idx]
                else:
                    pretty_print(MessageType.SUCCESS.name,"Invalid choice. Using first camera by default.")
                    selected_camera = cameras[0]
            except:
                pretty_print(MessageType.SUCCESS.name,"Invalid input. Using first camera by default.")
                selected_camera = cameras[0]

        cap = cv2.VideoCapture(selected_camera)
        time.sleep(2)
        return cap

    def trigger_alarm(self, times=3, interval=0.3):
        for _ in range(times):
            os.system("play -n synth 0.1 sine 5000 > /dev/null 2>&1")
            time.sleep(interval)


    def record_video(self, cap, duration_seconds=5, fps=20):
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        video_name = f"{self.detected_videos_dir}/detection_{timestamp_str}.avi"
                    
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
             
    def generate_face_name(self):
        """Generates a unique name for a new face"""
        self.face_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"Person_{self.face_counter:03d}_{timestamp}"

    def detect_and_recognize_faces(self, frame):
        """
        Detects and recognizes faces in the frame
        
        Returns:
            list: List of dictionaries with information about detected faces
        """
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []
        
        for _, (face_encoding, face_locations) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_locations
            
            
            if len(self.known_faces) > 0:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    
                    if matches[best_match_index] and confidence > 0.4:
                        
                        name = self.known_names[best_match_index]
                        is_new = False
                        pretty_print(MessageType.ALERT.name,f"👤 Recognized face: {name} (confidence: {confidence:.2f})")
                    else:
                        
                        name = self.generate_face_name()
                        is_new = True
                        self.known_faces.append(face_encoding)
                        self.known_names.append(name)
                        confidence = 1.0
                        pretty_print(MessageType.ALERT.name, f"🆕 New face detected: {name}")
                else:
                   
                    name = self.generate_face_name()
                    is_new = True
                    self.known_faces.append(face_encoding)
                    self.known_names.append(name)
                    confidence = 1.0
                    pretty_print(MessageType.ALERT.name, f"🆕 New face detected: {name}")
            else:
                
                name = self.generate_face_name()
                is_new = True
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                confidence = 1.0
                pretty_print(MessageType.ALERT.name, f"🆕 New face detected: {name}")
            
          
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(self.detected_faces_dir, filename)

            cv2.imwrite(filepath, frame)
            
            
            
            self.log_detection(name, is_new, confidence)
            
            # Add to detected faces list
            detected_faces.append({
                'name': name,
                'location': (left, top, right, bottom),
                'confidence': confidence,
                'is_new': is_new
            })
        
        
        if any(face['is_new'] for face in detected_faces):
            self.save_known_faces()
        
        return detected_faces
                
                

    def load_encondings(self):
        
        for filename in os.listdir(self.detected_faces_dir):

            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.detected_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image, model="large")

                if encodings:
                    self.known_faces.append(encodings[0])
                    self.known_names.append(filename.replace(".jpg", "").replace(".png", "").replace(".jpeg", "") )

        

    def preview_camera(self, cap):
        while True:
            _, frame = cap.read()
            
            cv2.putText(frame, "Press [Q] to start Detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 255), 2)
        
            cv2.imshow("Webcam Preview", frame)
          
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                break
            


    def get_frame(self, cap, scaling_factor=0.1):
        _, frame = cap.read()
       
        frame = cv2.resize(frame, None, fx=scaling_factor,
                        fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return frame



  