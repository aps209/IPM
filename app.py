import sys
import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. IMPORTACIONES ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from mediapipe.framework.formats import landmark_pb2 

# --- 2. FUNCIONES DE DISTANCIA ---
def get_normalized_distance(landmark1, landmark2) -> float:
    distance = np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
    return distance

# Distancia vertical (para ver si los dedos están estirados)
def get_vertical_distance(landmark1, landmark2) -> float:
    return abs(landmark1.y - landmark2.y)

# --- 3. NUEVA FUNCIÓN: DETECCIÓN DE GESTOS (Paso 7) ---
PINCH_THRESHOLD = 0.07 # Umbral para pellizco
POINT_THRESHOLD = 0.1 # Umbral para puntero (cuán estirado)

def detect_gesture(hand_landmarks_list):
    """
    Detecta un gesto (PELLIZCO o PUNTERO) en una mano.
    Devuelve: (Tipo de Gesto, Posición (x, y))
    """
    
    # --- Detección de PELLIZCO ---
    thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    pinch_distance = get_normalized_distance(thumb_tip, index_tip)
    
    if pinch_distance < PINCH_THRESHOLD:
        # Posición del pellizco: el centro entre los dos dedos
        pinch_x = (thumb_tip.x + index_tip.x) / 2
        pinch_y = (thumb_tip.y + index_tip.y) / 2
        return "PINCH", (pinch_x, pinch_y)

    # --- Detección de PUNTERO ---
    # Landmarks que usamos
    wrist = hand_landmarks_list[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP] # Base del índice
    
    # Puntas de los dedos
    middle_tip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_TIP]

    # 1. Índice estirado: La punta del índice (8) debe estar MÁS ARRIBA (menor 'y') que su base (5)
    if index_tip.y < index_mcp.y:
        # 2. Otros dedos encogidos: Sus puntas (12, 16, 20) deben estar MÁS ABAJO (mayor 'y') que la base del índice (5)
        if (middle_tip.y > index_mcp.y and 
            ring_tip.y > index_mcp.y and 
            pinky_tip.y > index_mcp.y):
            
            # Posición del puntero: la punta del dedo índice
            return "POINT", (index_tip.x, index_tip.y)

    return "NONE", (0, 0) # Ningún gesto detectado


# --- 4. FUNCIÓN: COMPOSICIÓN ALFA ---
def overlay_transparent(background, overlay, x, y):
    x = int(x)
    y = int(y)
    h_overlay, w_overlay = overlay.shape[:2]
    h_back, w_back = background.shape[:2]
    y1, y2 = max(0, y), min(h_back, y + h_overlay)
    x1, x2 = max(0, x), min(w_back, x + w_overlay)
    y1_overlay, y2_overlay = max(0, -y), max(0, -y) + (y2 - y1)
    x1_overlay, x2_overlay = max(0, -x), max(0, -x) + (x2 - x1)
    if (y2 - y1) <= 0 or (x2 - x1) <= 0: return
    roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_mask = cv2.merge([alpha, alpha, alpha])
    alpha_mask_inv = 1.0 - alpha_mask
    overlay_bgr = overlay_crop[:, :, :3]
    blended_roi = (alpha_mask * overlay_bgr + alpha_mask_inv * roi).astype(np.uint8)
    background[y1:y2, x1:x2] = blended_roi


# --- 5. CLASE: PIEZA DEL PUZZLE ---
class PuzzlePiece:
    def __init__(self, img_path, target_x, target_y, target_angle=0):
        self.original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
        self.original_img = cv2.resize(self.original_img, (100, 100), interpolation=cv2.INTER_AREA)
        self.h, self.w = self.original_img.shape[:2]
        
        self.x = np.random.randint(50, 200) # Posición central (x)
        self.y = np.random.randint(50, 200) # Posición central (y)
        self.angle = 0 
        
        self.target_x = target_x
        self.target_y = target_y
        self.target_angle = target_angle 
        
        self.is_held = False
        self.is_solved = False
        self.target_color = (150, 150, 150)
        self.snap_threshold_pos = 25
        self.snap_threshold_angle = 10 

    def draw(self, frame):
        target_color = (0, 255, 0) if self.is_solved else self.target_color
        cv2.rectangle(frame, (self.target_x, self.target_y), 
                      (self.target_x + self.w, self.target_y + self.h), 
                      target_color, 2)
        
        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((self.h * sin) + (self.w * cos)), int((self.h * cos) + (self.w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(self.original_img, M, (new_w, new_h))
        
        # Dibuja desde el centro (x, y)
        draw_x = self.x - (new_w // 2)
        draw_y = self.y - (new_h // 2)
        
        overlay_transparent(frame, rotated_img, draw_x, draw_y)

    def update_position(self, pos):
        if self.is_held:
            self.x, self.y = pos[0], pos[1] # pos es (x_pixel, y_pixel)

    # --- NUEVA FUNCIÓN (Paso 7) ---
    def rotate_piece(self):
        """Rota la pieza 90 grados en sentido horario."""
        if not self.is_solved:
            self.angle = (self.angle + 90) % 360
            print(f"¡ROTANDO! Nuevo ángulo: {self.angle}")

    def check_collision(self, pos) -> bool:
        (pinch_x, pinch_y) = pos
        # Comprobar si el punto (pos) está dentro del bounding box (caja) de la pieza
        # (Simplificado, ya que la pieza rota. Usamos el centro)
        dist = np.sqrt((pinch_x - self.x)**2 + (pinch_y - self.y)**2)
        return dist < self.w / 2 # Si toca cerca del centro

    def snap_to_target(self):
        dist_pos = np.sqrt((self.x - (self.target_x + self.w // 2))**2 + 
                           (self.y - (self.target_y + self.h // 2))**2)
        
        dist_angle = abs(self.angle - self.target_angle)
        dist_angle = min(dist_angle, 360 - dist_angle)

        if dist_pos < self.snap_threshold_pos and dist_angle < self.snap_threshold_angle:
            self.is_solved = True
            self.is_held = False
            self.x = self.target_x + self.w // 2
            self.y = self.target_y + self.h // 2
            self.angle = self.target_angle
            print("¡Pieza encajada (Posición y Ángulo)!")
        
        return self.is_solved

# --- 6. OPCIONES DE MEDIAPIPE ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# --- 7. CREAR PIEZA DE PUZZLE ---
piece1 = PuzzlePiece(img_path='images/piece1.png', 
                     target_x=400, 
                     target_y=200, 
                     target_angle=90)

# --- 8. BUCLE PRINCIPAL (MODIFICADO PARA GESTOS) ---
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0
    
    # Cooldown para rotación (evita 30 rotaciones por segundo)
    rotate_cooldown = 0
    COOLDOWN_FRAMES = 10 # Medio segundo aprox.

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms
        
        # Actualizar cooldown
        if rotate_cooldown > 0:
            rotate_cooldown -= 1

        # --- LÓGICA DE DETECCIÓN DE GESTOS (Paso 7) ---
        pinch_pos = None # Posición del pellizco (normalizada 0-1)
        point_pos = None # Posición del puntero (normalizada 0-1)

        if result.hand_landmarks:
            for hand_landmarks_list in result.hand_landmarks:
                
                # (Dibujar la mano)
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks_list
                ])
                mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                                          mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                          mp.solutions.drawing_styles.get_default_hand_connections_style())

                # Detectar gesto en esta mano
                gesture, pos = detect_gesture(hand_landmarks_list)
                
                if gesture == "PINCH":
                    pinch_pos = pos
                    # Dibujar círculo de pellizco
                    cv2.circle(frame, (int(pos[0]*w), int(pos[1]*h)), 10, (0, 255, 0), -1)

                elif gesture == "POINT":
                    point_pos = pos
                    # Dibujar círculo de puntero
                    cv2.circle(frame, (int(pos[0]*w), int(pos[1]*h)), 10, (0, 0, 255), -1)
        
        # --- LÓGICA DE ESTADO DEL JUEGO (Paso 7) ---
        if not piece1.is_solved:
            if not piece1.is_held:
                # --- ESTADO: Suelto ---
                # Si hay pellizco y colisiona, agarrar
                if pinch_pos:
                    pinch_pixel_pos = (int(pinch_pos[0]*w), int(pinch_pos[1]*h))
                    if piece1.check_collision(pinch_pixel_pos):
                        piece1.is_held = True
                        print("¡Agarrada!")
            else:
                # --- ESTADO: Agarrado ---
                if not pinch_pos:
                    # 1. No hay pellizco: Soltar
                    piece1.is_held = False
                    print("¡Soltada!")
                    piece1.snap_to_target()
                else:
                    # 2. Sigue el pellizco: Mover
                    pinch_pixel_pos = (int(pinch_pos[0]*w), int(pinch_pos[1]*h))
                    piece1.update_position(pinch_pixel_pos)
                    
                    # 3. ¿Hay un puntero "tocando" la pieza?
                    if point_pos and rotate_cooldown == 0:
                        point_pixel_pos = (int(point_pos[0]*w), int(point_pos[1]*h))
                        if piece1.check_collision(point_pixel_pos):
                            piece1.rotate_piece()
                            rotate_cooldown = COOLDOWN_FRAMES # Activar cooldown

        piece1.draw(frame)
        cv2.imshow("Hand Landmarker - Paso 7 (Pellizco y Puntero)", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27: # ESC
            break
        
    cap.release()
    cv2.destroyAllWindows()