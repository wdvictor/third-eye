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
        Sistema completo de reconhecimento facial
        
        Args:
            faces_dir: Diretório para salvar imagens das faces
            database_dir: Diretório para salvar banco de dados de faces
        """
        self.faces_dir = faces_dir
        self.database_dir = database_dir
        self.log_file = "face_detection_log.json"
        
        # Criar diretórios se não existirem
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(database_dir, exist_ok=True)
        
        # Banco de dados de faces conhecidas
        self.known_faces = []
        self.known_names = []
        self.face_counter = 0
        
        # Carregar faces conhecidas
        self.load_known_faces()
        
        # Inicializar detectores
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print(f"✅ Sistema inicializado!")
        print(f"📁 Faces salvas em: {faces_dir}")
        print(f"🗃️  Database em: {database_dir}")
        print(f"📋 Log em: {self.log_file}")
    
    def load_known_faces(self):
        """Carrega faces conhecidas do banco de dados"""
        database_file = os.path.join(self.database_dir, "faces_database.pkl")
        
        if os.path.exists(database_file):
            try:
                with open(database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                    self.face_counter = data.get('counter', 0)
                print(f"📚 Carregadas {len(self.known_faces)} faces conhecidas")
            except Exception as e:
                print(f"⚠️  Erro ao carregar database: {e}")
        else:
            print("🆕 Criando novo banco de dados de faces")
    
    def save_known_faces(self):
        """Salva faces conhecidas no banco de dados"""
        database_file = os.path.join(self.database_dir, "faces_database.pkl")
        
        try:
            data = {
                'faces': self.known_faces,
                'names': self.known_names,
                'counter': self.face_counter
            }
            with open(database_file, 'wb') as f:
                pickle.dump(data, f)
            print("💾 Database salvo com sucesso")
        except Exception as e:
            print(f"❌ Erro ao salvar database: {e}")
    
    def generate_face_name(self):
        """Gera um nome único para uma nova face"""
        self.face_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"Pessoa_{self.face_counter:03d}_{timestamp}"
    
    def log_detection(self, name, is_new_face=False, confidence=0.0):
        """Registra detecção no log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "is_new_face": is_new_face,
            "confidence": confidence
        }
        
        # Carregar log existente
        log_data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = []
        
        # Adicionar nova entrada
        log_data.append(log_entry)
        
        # Salvar log
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"❌ Erro ao salvar log: {e}")
    
    def detect_and_recognize_faces(self, frame):
        print('detect_and_recognize_faces')
        """
        Detecta e reconhece faces no frame
        
        Returns:
            list: Lista de dicionários com informações das faces detectadas
        """
        # Converter para RGB (face_recognition usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar faces usando face_recognition (mais preciso)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []
        print('detect_and_recognize_faces 2')
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_location
            
            # Verificar se é uma face conhecida
            if len(self.known_faces) > 0:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    
                    if matches[best_match_index] and confidence > 0.4:
                        # Face conhecida
                        name = self.known_names[best_match_index]
                        is_new = False
                        print(f"👤 Face reconhecida: {name} (confiança: {confidence:.2f})")
                    else:
                        # Nova face
                        name = self.generate_face_name()
                        is_new = True
                        self.known_faces.append(face_encoding)
                        self.known_names.append(name)
                        confidence = 1.0
                        print(f"🆕 Nova face detectada: {name}")
                else:
                    # Primeira face no sistema
                    name = self.generate_face_name()
                    is_new = True
                    self.known_faces.append(face_encoding)
                    self.known_names.append(name)
                    confidence = 1.0
                    print(f"🆕 Primeira face detectada: {name}")
            else:
                # Primeira face no sistema
                name = self.generate_face_name()
                is_new = True
                self.known_faces.append(face_encoding)
                self.known_names.append(name)
                confidence = 1.0
                print(f"🆕 Primeira face detectada: {name}")
            
            # Salvar imagem da face
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(self.faces_dir, filename)
                cv2.imwrite(filepath, face_img)
                print(f"💾 Face salva: {filename}")
            
            # Registrar no log
            self.log_detection(name, is_new, confidence)
            
            # Adicionar à lista de faces detectadas
            detected_faces.append({
                'name': name,
                'location': (left, top, right, bottom),
                'confidence': confidence,
                'is_new': is_new
            })
        
        # Salvar database se houver novas faces
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
    """Detecta movimento baseado na máscara"""
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
    """Desenha informações das faces no frame"""
    for face in faces:
        left, top, right, bottom = face['location']
        name = face['name']
        confidence = face['confidence']
        is_new = face['is_new']
        
        # Cor do retângulo (verde para conhecida, azul para nova)
        color = (0, 255, 255) if is_new else (0, 255, 0)
        
        # Desenhar retângulo
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Texto com nome e confiança
        label = f"{name} ({confidence:.2f})"
        if is_new:
            label += " [NOVA]"
        
        # Fundo para o texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width, top), color, -1)
        
        # Texto
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

if __name__ == '__main__':
    # Verificar se face_recognition está instalado
    try:
        import face_recognition
    except ImportError:
        print("❌ Erro: face_recognition não está instalado!")
        print("📦 Para instalar: pip install face_recognition")
        print("⚠️  No Windows, você pode precisar instalar: pip install cmake")
        exit()
    
    # Inicializar sistema de reconhecimento facial
    face_system = FaceRecognitionSystem()
    
    # Inicializar captura de vídeo
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        exit()
    
    # # Configurar câmera
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Inicializar subtrator de fundo
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
    
    # Variáveis de controle
    last_motion_time = 0
    motion_cooldown = 3  # Cooldown para detecção de movimento
    face_detection_cooldown = 5  # Cooldown para detecção de faces
    last_face_detection = 0
    frame_count = 0
    learning_frames = 30
    
    print("\n🎥 SISTEMA DE RECONHECIMENTO FACIAL ATIVO!")
    print("=" * 60)
    print("📋 Instruções:")
    print("   • Aguarde calibração do fundo")
    print("   • Faça um movimento para ativar detecção facial")
    print("   • ESC ou 'q' para sair")
    print("   • 'r' para recalibrar")
    print("   • 's' para salvar database manualmente")
    print("=" * 60)
    
    while True:
        try:
            frame = get_frame(cap, 0.8)
            
            if frame is None:
                print("❌ Erro ao capturar frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Taxa de aprendizado adaptativa
            learning_rate = 0.1 if frame_count < learning_frames else 0.005
            
            # Aplicar subtração de fundo
            mask = bgSubtractor.apply(frame, learningRate=learning_rate)
            
            # Sistema ativo apenas após calibração
            if frame_count > learning_frames:
                # Detectar movimento
                has_motion, contours = detect_motion(mask, min_area=1000)
                
                # Se há movimento e passou do cooldown
                if has_motion and (current_time - last_motion_time) > motion_cooldown:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"🚨 MOVIMENTO DETECTADO [{timestamp}] - Iniciando detecção facial...")
                    last_motion_time = current_time
                    
                    # Detectar faces se passou do cooldown de face
                    if (current_time - last_face_detection) > face_detection_cooldown:
                        print("🔍 Analisando faces...")
                        detected_faces = face_system.detect_and_recognize_faces(frame)
                        
                        if detected_faces:
                            print(f"✅ {len(detected_faces)} face(s) processada(s)")
                            # Desenhar informações das faces
                            draw_face_info(frame, detected_faces)
                        else:
                            print("⚠️  Movimento detectado, mas nenhuma face encontrada")
                        
                        last_face_detection = current_time
                    else:
                        remaining = face_detection_cooldown - (current_time - last_face_detection)
                        print(f"⏱️  Aguarde {remaining:.1f}s para nova detecção facial")
                
                # Desenhar contornos de movimento
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Status no frame
            if frame_count <= learning_frames:
                progress = int((frame_count / learning_frames) * 100)
                cv2.putText(frame, f"Calibrando... {progress}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "SISTEMA ATIVO", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Info do banco de dados
                cv2.putText(frame, f"Faces conhecidas: {len(face_system.known_faces)}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Exibir frames
            cv2.imshow('Sistema de Reconhecimento Facial', frame)
            cv2.imshow('Detector de Movimento', mask)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC ou 'q'
                break
            elif key == ord('r'):  # Reset
                bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
                frame_count = 0
                print("🔄 Sistema recalibrado")
            elif key == ord('s'):  # Salvar database
                face_system.save_known_faces()
                print("💾 Database salvo manualmente")
                
        except Exception as erro:
            print(f"❌ Erro: {erro}")
            break
    
    # Finalizar
    face_system.save_known_faces()
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema finalizado com sucesso!")
    print(f"📊 Total de faces conhecidas: {len(face_system.known_faces)}")
    print(f"📁 Imagens salvas em: {face_system.faces_dir}")
    print(f"📋 Log disponível em: {face_system.log_file}")