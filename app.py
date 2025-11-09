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

# Necesitamos este formato para que 'draw_landmarks' funcione
from mediapipe.framework.formats import landmark_pb2 


# --- 2. FUNCIÓN DE DISTANCIA (Paso 2) ---
def get_normalized_distance(landmark1, landmark2) -> float:
    distance = np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
    return distance

# --- 3. NUEVA CLASE: PIEZA DEL PUZZLE (Paso 3) ---
class PuzzlePiece:
    def __init__(self, x, y, target_x, target_y, size=50):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.size = size
        self.is_held = False
        self.is_solved = False
        self.color = (255, 0, 0) # Azul (sin resolver)
        self.solved_color = (0, 255, 0) # Verde (resuelto)
        self.target_color = (150, 150, 150) # Gris (meta)
        self.snap_threshold = 20

    def draw(self, frame):
        cv2.rectangle(frame, (self.target_x, self.target_y), 
                      (self.target_x + self.size, self.target_y + self.size), 
                      self.target_color, 2)
        
        current_color = self.solved_color if self.is_solved else self.color
        cv2.rectangle(frame, (self.x, self.y), 
                      (self.x + self.size, self.y + self.size), 
                      current_color, -1)

    def update_position(self, x, y):
        if self.is_held:
            self.x = x - self.size // 2
            self.y = y - self.size // 2

    def check_collision(self, pinch_x, pinch_y) -> bool:
        return (self.x < pinch_x < self.x + self.size) and \
               (self.y < pinch_y < self.y + self.size)

    def snap_to_target(self):
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        target_center_x = self.target_x + self.size // 2
        target_center_y = self.target_y + self.size // 2
        
        dist = np.sqrt((center_x - target_center_x)**2 + (center_y - target_center_y)**2)
        
        if dist < self.snap_threshold:
            self.is_solved = True
            self.is_held = False
            self.x = self.target_x
            self.y = self.target_y
            print("¡Pieza encajada!")
        
        return self.is_solved

# --- 4. OPCIONES DE MEDIAPIPE ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# --- 5. CREAR NUESTRA PIEZA DE PUZZLE ---
piece1 = PuzzlePiece(x=100, y=100, target_x=400, target_y=200, size=50)

# --- 6. BUCLE PRINCIPAL ---
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0

    PINCH_THRESHOLD = 0.07 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms

        is_pinching = False
        pinch_x, pinch_y = 0, 0

        # --- LÓGICA DE DETECCIÓN ---
        if result.hand_landmarks:
            for hand_landmarks_list in result.hand_landmarks:
                
                # --- INICIO DE LA CORRECCIÓN ---
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                
                # Convertimos la lista de Python (Formato A) a la lista Protobuf (Formato B)
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    ) for landmark in hand_landmarks_list
                ])
                # --- FIN DE LA CORRECCIÓN ---

                # Dibujar usando el objeto 'proto'
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks_proto, # <- Usamos el objeto corregido
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

                # Lógica de pellizco
                thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP]
                # --- CORRECCIÓN DE ERROR TIPOGRÁFICO AQUÍ ---
                index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinch_distance = get_normalized_distance(thumb_tip, index_tip)

                if pinch_distance < PINCH_THRESHOLD:
                    is_pinching = True
                    pinch_x = int((thumb_tip.x + index_tip.x) * w / 2)
                    pinch_y = int((thumb_tip.y + index_tip.y) * h / 2)
                    
                    cv2.circle(frame, (pinch_x, pinch_y), 10, (0, 255, 0), -1)
                    break 
        
        # --- LÓGICA DE ESTADO DEL JUEGO ---
        if not piece1.is_solved:
            if is_pinching:
                if piece1.is_held:
                    piece1.update_position(pinch_x, pinch_y)
                else:
                    if piece1.check_collision(pinch_x, pinch_y):
                        piece1.is_held = True
                        print("¡Agarrada!")
            else:
                if piece1.is_held:
                    piece1.is_held = False
                    print("¡Soltada!")
                    piece1.snap_to_target()

        # --- DIBUJAR LA PIEZA (SIEMPRE) ---
        piece1.draw(frame)

        cv2.imshow("Hand Landmarker - Paso 3 (Agarrar)", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()