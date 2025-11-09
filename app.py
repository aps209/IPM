import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import random 

# --- 1. IMPORTACIONES ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from mediapipe.framework.formats import landmark_pb2 

# --- CONSTANTES DEL JUEGO ---
PIECE_SIZE = 100 
GRID_COLOR = (100, 100, 100) 
SPAWN_X_LIMIT = 250 

# --- 2. FUNCIONES DE GESTOS Y DISTANCIA ---
def get_normalized_distance(landmark1, landmark2) -> float:
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

PINCH_THRESHOLD = 0.07 
def detect_gesture(hand_landmarks_list):
    thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinch_distance = get_normalized_distance(thumb_tip, index_tip)
    if pinch_distance < PINCH_THRESHOLD:
        pos = ((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)
        return "PINCH", pos

    index_mcp = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_TIP]
    if index_tip.y < index_mcp.y and (middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y and pinky_tip.y > index_mcp.y):
        return "POINT", (index_tip.x, index_tip.y)

    p0, p5, p17 = hand_landmarks_list[0], hand_landmarks_list[5], hand_landmarks_list[17]
    pos = ((p0.x + p5.x + p17.x) / 3, (p0.y + p5.y + p17.y) / 3)
    return "HAND", pos

# --- 3. FUNCIÓN: COMPOSICIÓN ALFA ---
def overlay_transparent(background, overlay, x, y):
    x, y = int(x), int(y)
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

# --- 4. FUNCIÓN: DIBUJAR CUADRÍCULA ---
def draw_grid(canvas, grid_origin, grid_size, piece_size):
    rows, cols = grid_size
    (ox, oy) = grid_origin
    for r in range(rows + 1):
        y = oy + r * piece_size
        cv2.line(canvas, (ox, y), (ox + cols * piece_size, y), GRID_COLOR, 2)
    for c in range(cols + 1):
        x = ox + c * piece_size
        cv2.line(canvas, (x, oy), (x, oy + rows * piece_size), GRID_COLOR, 2)

# --- 5. CLASE: PIEZA DEL PUZZLE ---
class PuzzlePiece:
    def __init__(self, img_path, target_x, target_y, target_angle, initial_angle):
        self.original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
        self.original_img = cv2.resize(self.original_img, (PIECE_SIZE, PIECE_SIZE), interpolation=cv2.INTER_AREA)
        self.h, self.w = self.original_img.shape[:2]
        self.x = np.random.randint(self.w // 2, SPAWN_X_LIMIT - self.w // 2)
        self.y = np.random.randint(self.h // 2, 480 - self.h // 2) 
        self.angle = initial_angle
        self.target_x = target_x
        self.target_y = target_y
        self.target_angle = target_angle 
        self.is_held = False
        self.is_solved = False
        self.snap_threshold_pos = 25
        self.snap_threshold_angle = 10 

    def draw(self, frame):
        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((self.h * sin) + (self.w * cos)), int((self.h * cos) + (self.w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated_img = cv2.warpAffine(self.original_img, M, (new_w, new_h))
        draw_x = self.x - (new_w // 2)
        draw_y = self.y - (new_h // 2)
        overlay_transparent(frame, rotated_img, draw_x, draw_y)

    def update_position(self, pos):
        if self.is_held:
            self.x, self.y = pos[0], pos[1]

    def rotate_piece(self):
        if not self.is_solved and self.is_held:
            self.angle = (self.angle + 90) % 360
            print(f"¡ROTANDO! Nuevo ángulo: {self.angle}")

    def check_collision(self, pos) -> bool:
        (pinch_x, pinch_y) = pos
        dist = np.sqrt((pinch_x - self.x)**2 + (pinch_y - self.y)**2)
        return dist < self.w / 2 

    def snap_to_target(self):
        target_center_x = self.target_x + self.w // 2
        target_center_y = self.target_y + self.h // 2
        dist_pos = np.sqrt((self.x - target_center_x)**2 + (self.y - target_center_y)**2)
        dist_angle = abs(self.angle - self.target_angle)
        dist_angle = min(dist_angle, 360 - dist_angle)
        if dist_pos < self.snap_threshold_pos and dist_angle < self.snap_threshold_angle:
            self.is_solved = True
            self.is_held = False
            self.x = target_center_x 
            self.y = target_center_y
            self.angle = self.target_angle
            print(f"¡Pieza ({self.target_x}, {self.target_y}) encajada!")
        return self.is_solved

# --- 6. DEFINICIÓN DE NIVELES (Paso 11) ---
PUZZLE_DEFINITIONS = {
    "pato": {
        "folder": "images/pato/",
        "grid_size": (2, 2), # 2 filas, 2 columnas
        "difficulty": 1 # Multiplicador de puntuación (Fácil)
    },
    "gato": {
        "folder": "images/gato/", # <-- AÑADE TUS IMÁGENES AQUÍ
        "grid_size": (3, 2), # 3 filas, 2 columnas (6 piezas)
        "difficulty": 2 # (Medio)
    }
    # (Añade aquí tu puzzle de 7 piezas como "dificil")
}

# --- 7. FUNCIÓN: CARGAR PUZZLE ---
def load_puzzle(puzzle_name, grid_origin):
    if puzzle_name not in PUZZLE_DEFINITIONS:
        raise ValueError(f"Puzzle '{puzzle_name}' no definido.")
    
    definition = PUZZLE_DEFINITIONS[puzzle_name]
    folder = definition["folder"]
    rows, cols = definition["grid_size"]
    (ox, oy) = grid_origin
    
    pieces = []
    for r in range(rows):
        for c in range(cols):
            img_path = f"{folder}piece_{r}_{c}.png"
            target_x = ox + c * PIECE_SIZE
            target_y = oy + r * PIECE_SIZE
            target_angle = 0 
            initial_angle = random.choice([0, 90, 180, 270])
            piece = PuzzlePiece(img_path, target_x, target_y, target_angle, initial_angle)
            pieces.append(piece)
    
    # Devuelve también la definición para usarla en la puntuación
    return pieces, definition["grid_size"], definition["difficulty"]


# --- 8. OPCIONES DE MEDIAPIPE ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# --- 9. BUCLE PRINCIPAL (MODIFICADO) ---
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0
    rotate_cooldown = 0
    COOLDOWN_FRAMES = 10 

    ret, frame = cap.read()
    if not ret: sys.exit(1)
    h, w, _ = frame.shape
    
    # --- Cargar el puzzle ALEATORIO (Paso 11) ---
    GRID_ORIGIN = (w - (PIECE_SIZE * 3), h // 2 - (PIECE_SIZE)) # Centrar cuadrícula 
    
    # --- Lógica de selección aleatoria ---
    puzzle_names = list(PUZZLE_DEFINITIONS.keys())
    loaded_puzzle_name = random.choice(puzzle_names)
    print(f"--- Cargando puzzle aleatorio: {loaded_puzzle_name} ---")
    
    try:
        puzzle_pieces, grid_size, difficulty = load_puzzle(loaded_puzzle_name, GRID_ORIGIN)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Asegúrate de que la carpeta '{loaded_puzzle_name}' y sus piezas existen.")
        sys.exit(1)
    
    # --- Variables de estado del juego (Paso 11) ---
    held_piece = None
    game_won = False
    start_time = time.time() # <-- INICIAR CRONÓMETRO
    elapsed_time = 0
    final_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        canvas = np.full((h, w, 3), (220, 245, 245), dtype="uint8")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms
        
        if rotate_cooldown > 0:
            rotate_cooldown -= 1

        pinch_pos, point_pos = None, None 

        if result.hand_landmarks:
            for hand_landmarks_list in result.hand_landmarks:
                gesture, pos = detect_gesture(hand_landmarks_list)
                cursor_pos_x = int(pos[0] * w)
                cursor_pos_y = int(pos[1] * h)
                cursor_pos_x = w - cursor_pos_x 

                if gesture == "PINCH":
                    pinch_pos = (pos[0], pos[1]) 
                    cv2.circle(canvas, (cursor_pos_x, cursor_pos_y), 15, (0, 255, 0), -1)
                elif gesture == "POINT":
                    point_pos = (pos[0], pos[1])
                    cv2.circle(canvas, (cursor_pos_x, cursor_pos_y), 15, (255, 0, 0), -1)
                elif gesture == "HAND":
                    cv2.circle(canvas, (cursor_pos_x, cursor_pos_y), 10, (0, 0, 255), -1)
        
        # --- LÓGICA DE JUEGO ---
        if not game_won:
            if held_piece is None:
                if pinch_pos:
                    pinch_pixel_x, pinch_pixel_y = int((1.0 - pinch_pos[0]) * w), int(pinch_pos[1] * h)
                    for piece in reversed(puzzle_pieces):
                        if not piece.is_solved and piece.check_collision((pinch_pixel_x, pinch_pixel_y)):
                            held_piece = piece
                            held_piece.is_held = True
                            break 
            else:
                if not pinch_pos:
                    held_piece.is_held = False
                    held_piece.snap_to_target()
                    held_piece = None
                else:
                    pinch_pixel_x, pinch_pixel_y = int((1.0 - pinch_pos[0]) * w), int(pinch_pos[1] * h)
                    held_piece.update_position((pinch_pixel_x, pinch_pixel_y))
                    
                    if point_pos and rotate_cooldown == 0:
                        point_pixel_x, point_pixel_y = int((1.0 - point_pos[0]) * w), int(point_pos[1] * h)
                        if held_piece.check_collision((point_pixel_x, point_pixel_y)):
                            held_piece.rotate_piece()
                            rotate_cooldown = COOLDOWN_FRAMES 

        # --- DIBUJADO ---
        draw_grid(canvas, GRID_ORIGIN, grid_size, PIECE_SIZE)
        for piece in puzzle_pieces:
            if piece is not held_piece:
                piece.draw(canvas)
        if held_piece:
            held_piece.draw(canvas)

        # --- LÓGICA DE VICTORIA Y PUNTUACIÓN (Paso 11) ---
        if not game_won:
            if all(p.is_solved for p in puzzle_pieces):
                game_won = True
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Puntuación: 10000 * dificultad / segundos
                final_score = (10000 * difficulty) / elapsed_time 
                print(f"¡HAS GANADO! Tiempo: {elapsed_time:.2f}s, Puntuación: {final_score:.0f}")
                
        if game_won:
            # Mostrar pantalla de victoria
            cv2.putText(canvas, "HAS GANADO", (w//2 - 190, h//2 - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            cv2.putText(canvas, f"Tiempo: {elapsed_time:.2f}s", (w//2 - 150, h//2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(canvas, f"Puntuacion: {final_score:.0f}", (w//2 - 160, h//2 + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        else:
            # Mostrar tiempo actual
            current_elapsed = time.time() - start_time
            cv2.putText(canvas, f"Tiempo: {current_elapsed:.1f}", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Juego de Puzzle IPM - Paso 11", canvas)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()