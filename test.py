import cv2
import numpy as np
import time

def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.resize(frame, None, fx=scaling_factor,
                      fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def detect_motion(mask, min_area=500):
    """
    Detecta movimento baseado na máscara de subtração de fundo
    
    Args:
        mask: Máscara binária da subtração de fundo
        min_area: Área mínima para considerar como movimento válido
    
    Returns:
        bool: True se movimento foi detectado, False caso contrário
        list: Lista de contornos dos objetos em movimento
    """
    # Aplicar operações morfológicas para reduzir ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área mínima
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            valid_contours.append(contour)
    
    return len(valid_contours) > 0, valid_contours

if __name__ == '__main__':
    # Inicializar captura de vídeo
    cap = cv2.VideoCapture(0)
    
    # Verificar se a câmera foi aberta corretamente
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        exit()
    
    # Configurar propriedades da câmera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Inicializar o subtrator de fundo
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
    
    # Variáveis para controle de detecção
    motion_detected = False
    last_motion_time = 0
    motion_cooldown = 2  # Segundos entre detecções para evitar spam
    frame_count = 0
    learning_frames = 30  # Frames para aprender o fundo
    
    print("🎥 Detector de Movimento Iniciado!")
    print("📋 Aguarde alguns segundos para calibrar o fundo...")
    print("⌨️  Pressione 'ESC' para sair ou 'q' para quit")
    print("=" * 50)
    
    while True:
        try:
            # Capturar frame
            frame = get_frame(cap, 0.7)
            
            # Verificar se o frame foi capturado corretamente
            if frame is None:
                print("Erro: Não foi possível capturar o frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Usar taxa de aprendizado mais alta nos primeiros frames
            learning_rate = 0.1 if frame_count < learning_frames else 0.005
            
            # Aplicar subtração de fundo
            mask = bgSubtractor.apply(frame, learningRate=learning_rate)
            
            # Detectar movimento apenas após calibração inicial
            if frame_count > learning_frames:
                has_motion, contours = detect_motion(mask, min_area=800)
                
                # Se movimento foi detectado e passou do cooldown
                if has_motion and (current_time - last_motion_time) > motion_cooldown:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"🚨 MOVIMENTO DETECTADO! [{timestamp}] - {len(contours)} objeto(s)")
                    last_motion_time = current_time
                    motion_detected = True
                
                # Desenhar contornos no frame (opcional para visualização)
                if contours:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "MOVIMENTO", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Mostrar progresso da calibração
                progress = int((frame_count / learning_frames) * 100)
                cv2.putText(frame, f"Calibrando... {progress}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Status no frame
            status_text = "MONITORANDO" if frame_count > learning_frames else "CALIBRANDO"
            status_color = (0, 255, 0) if frame_count > learning_frames else (0, 165, 255)
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Exibir frames
            cv2.imshow('Detector de Movimento', frame)
            cv2.imshow('Mascara de Movimento', mask)
            
            # Verificar teclas pressionadas
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC ou 'q'
                break
            elif key == ord('r'):  # Reset calibração
                bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
                frame_count = 0
                print("🔄 Recalibrando detector...")
                
        except Exception as erro:
            print(f"❌ Erro inesperado: {erro}")
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detector finalizado. Recursos liberados com sucesso")