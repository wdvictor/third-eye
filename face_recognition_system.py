import cv2
import numpy as np
import time
import os
import pickle
import json
from datetime import datetime
import face_recognition

class FaceRecognitionSystem:
    def __init__(self, faces_dir="detected_faces", database_dir="face_database"):
        """
        Complete facial recognition system
        
        Args:
            faces_dir: Directory to save face images
            database_dir: Directory to save face database
        """
        self.faces_dir = faces_dir
        self.database_dir = database_dir
        self.log_file = "face_detection_log.json"
        
        
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(database_dir, exist_ok=True)
        
        
        self.known_faces = []
        self.known_names = []
        self.face_counter = 0
        
        
        self.load_known_faces()
        
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print(f"✅ System initialized!")
        print(f"📁 Faces saved in: {faces_dir}")
        print(f"🗃️  Database in: {database_dir}")
        print(f"📋 Log in: {self.log_file}")
    
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
                print(f"📚 Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"⚠️  Error loading database: {e}")
        else:
            print("🆕 Creating new face database")
    
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
            print("💾 Database saved successfully")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
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
        
        # Load existing log
        log_data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = []
        
        # Add new entry
        log_data.append(log_entry)
        
        # Save log
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving log: {e}")
    
    def detect_and_recognize_faces(self, frame):
        print('detect_and_recognize_faces')
        """
        Detects and recognizes faces in the frame
        
        Returns:
            list: List of dictionaries with information about detected faces
        """
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using face_recognition (more accurate)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []
        print('detect_and_recognize_faces 2')
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_location
            
            # Check if it is a known face
            if len(self.known_faces) > 0:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    
                    if matches[best_match_index] and confidence > 0.4:
                        # Known face
                        name = self.known_names[best_match_index]
                        is_new = False
                        print(f"👤 Recognized face: {name} (confidence: {confidence:.2f})")
                    else:
                        # New face
                        name = self.generate_face_name()
                        is_new = True
                        self.known_faces.append(face_encoding)
                        self.known_names.append(name)
                        confidence = 1.0
                        print(f"🆕 New face detected: {name}")
                else:
                    # First face in the system
                    name = self.generate_face_name()
                    is_new = True
                    self.known_faces.append(face_encoding)
                    self.known_names.append(name)
                    confidence = 1.0
                    print(f"🆕 First face detected: {name}")
            else:
                # First face in the system
                name = self.generate_face_name()
                is_new = True
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                confidence = 1.0
                print(f"🆕 First face detected: {name}")
            
            # Save face image
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(self.faces_dir, filename)
                cv2.imwrite(filepath, face_img)
                print(f"💾 Face saved: {filename}")
            
            # Log detection
            self.log_detection(name, is_new, confidence)
            
            # Add to detected faces list
            detected_faces.append({
                'name': name,
                'location': (left, top, right, bottom),
                'confidence': confidence,
                'is_new': is_new
            })
        
        # Save database if there are new faces
        if any(face['is_new'] for face in detected_faces):
            self.save_known_faces()
        
        return detected_faces

def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.resize(frame, None, fx=scaling_factor,
                      fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def detect_motion(mask, min_area=800):
    """Detects motion based on the mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            valid_contours.append(contour)
    
    return len(valid_contours) > 0, valid_contours

def draw_face_info(frame, faces):
    """Draws face information on the frame"""
    for face in faces:
        left, top, right, bottom = face['location']
        name = face['name']
        confidence = face['confidence']
        is_new = face['is_new']
        
        # Rectangle color (green for known, blue for new)
        color = (0, 255, 255) if is_new else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Text with name and confidence
        label = f"{name} ({confidence:.2f})"
        if is_new:
            label += " [NEW]"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width, top), color, -1)
        
        # Text
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

if __name__ == '__main__':
    # Check if face_recognition is installed
    try:
        import face_recognition
    except ImportError:
        print("❌ Error: face_recognition is not installed!")
        print("📦 To install: pip install face_recognition")
        print("⚠️  On Windows, you may need to install: pip install cmake")
        exit()
    

    face_system = FaceRecognitionSystem()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        exit()
    
    # Initialize background subtractor
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
    
    # Control variables
    last_motion_time = 0
    motion_cooldown = 3  # Cooldown for motion detection
    face_detection_cooldown = 5  # Cooldown for face detection
    last_face_detection = 0
    frame_count = 0
    learning_frames = 30
    
    print("\n🎥 FACIAL RECOGNITION SYSTEM ACTIVE!")
    print("=" * 60)
    print("📋 Instructions:")
    print("   • Wait for background calibration")
    print("   • Make a movement to activate facial detection")
    print("   • ESC or 'q' to exit")
    print("   • 'r' to recalibrate")
    print("   • 's' to save database manually")
    print("=" * 60)
    
    while True:
        try:
            frame = get_frame(cap, 1)
            
            if frame is None:
                print("❌ Error capturing frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Adaptive learning rate
            learning_rate = 0.1 if frame_count < learning_frames else 0.005
            
            # Apply background subtraction
            mask = bgSubtractor.apply(frame, learningRate=learning_rate)
            
            # System active only after calibration
            if frame_count > learning_frames:
                # Detect motion
                has_motion, contours = detect_motion(mask, min_area=1000)
                
                # If there is motion and cooldown has passed
                if has_motion and (current_time - last_motion_time) > motion_cooldown:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"🚨 MOTION DETECTED [{timestamp}] - Starting facial detection...")
                    last_motion_time = current_time
                    
                    # Detect faces if face detection cooldown has passed
                    if (current_time - last_face_detection) > face_detection_cooldown:
                        print("🔍 Analyzing faces...")
                        detected_faces = face_system.detect_and_recognize_faces(frame)
                        
                        if detected_faces:
                            print(f"✅ {len(detected_faces)} face(s) processed")
                            # Draw face information
                            draw_face_info(frame, detected_faces)
                        else:
                            print("⚠️  Motion detected, but no face found")
                        
                        last_face_detection = current_time
                    else:
                        remaining = face_detection_cooldown - (current_time - last_face_detection)
                        print(f"⏱️  Wait {remaining:.1f}s for next facial detection")
                
                # Draw motion contours
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Status on frame
            if frame_count <= learning_frames:
                progress = int((frame_count / learning_frames) * 100)
                cv2.putText(frame, f"Calibrating... {progress}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "SYSTEM ACTIVE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Database info
                cv2.putText(frame, f"Known faces: {len(face_system.known_faces)}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frames
            cv2.imshow('Facial Recognition System', frame)
            cv2.imshow('Motion Detector', mask)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('r'):  # Reset
                bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
                frame_count = 0
                print("🔄 System recalibrated")
            elif key == ord('s'):  # Save database
                face_system.save_known_faces()
                print("💾 Database saved manually")
                
        except Exception as erro:
            print(f"❌ Error: {erro}")
            break
    
    # Finish
    face_system.save_known_faces()
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ System finished successfully!")
    print(f"📊 Total known faces: {len(face_system.known_faces)}")
    print(f"📁 Images saved in: {face_system.faces_dir}")
    print(f"📋 Log available at: {face_system.log_file}")