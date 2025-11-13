import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import random 
import pygame 

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

# --- CONSTANTES DE AYUDA ---
HELP_DELAY_SECONDS = 10 
HELP_SIZE = 300 
HELP_POSITION = (10, 10) 

# --- CONSTANTES DE RECOMPENSA (NUEVO) ---
FULL_GHOST_DURATION_SECONDS = 5.0 # 5 segundos para la pista del 25%

# --- CONSTANTES DE AUDIO ---
MENU_MUSIC_PATH = "music/menu.mp3" 
GAME_MUSIC_PATH = "music/game.mp3" 
VICTORY_MUSIC_PATH = "music/victory.mp3"

# --- 2. FUNCIONES DE GESTOS Y DISTANCIA ---
def get_normalized_distance(landmark1, landmark2) -> float:
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

PINCH_THRESHOLD = 0.07 
def detect_gesture(hand_landmarks_list):
    # Gestos se definen por las puntas de los dedos
    thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_TIP]
    
    # Nudillos (MCP - base de los dedos)
    index_mcp = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks_list[mp_hands.HandLandmark.PINKY_MCP] 

    # --- Detección de PELLIZCO ---
    pinch_distance = get_normalized_distance(thumb_tip, index_tip)
    if pinch_distance < PINCH_THRESHOLD:
        pos = ((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)
        return "PINCH", pos

    # --- Detección de PUNTERO ---
    if index_tip.y < index_mcp.y and (middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y):
        return "POINT", (index_tip.x, index_tip.y)
    
    # --- Detección de PUÑO (FIST) ---
    if (index_tip.y > index_mcp.y and 
        middle_tip.y > middle_mcp.y and 
        ring_tip.y > ring_mcp.y and 
        pinky_tip.y > pinky_mcp.y):
        fist_x = (index_mcp.x + middle_mcp.x) / 2
        fist_y = (index_mcp.y + middle_mcp.y) / 2
        return "FIST", (fist_x, fist_y)

    # --- Gesto por defecto: MANO ---
    p0, p5, p17 = hand_landmarks_list[0], hand_landmarks_list[5], hand_landmarks_list[17]
    pos = ((p0.x + p5.x + p17.x) / 3, (p0.y + p5.y + p17.y) / 3)
    return "HAND", pos

# --- 3. FUNCIÓN: COMPOSICIÓN ALFA ---
def overlay_transparent(background, overlay, x, y, opacity=1.0):
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
    
    alpha = (overlay_crop[:, :, 3] / 255.0) * opacity 
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
    def __init__(self, image_slice, target_x, target_y, target_angle, initial_angle):
        
        self.original_img = cv2.resize(image_slice, (PIECE_SIZE, PIECE_SIZE), interpolation=cv2.INTER_AREA)
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

    def draw_ghost(self, frame):
        draw_x = self.target_x
        draw_y = self.target_y
        img_to_draw = cv2.resize(self.original_img, (self.w, self.h))
        overlay_transparent(frame, img_to_draw, draw_x, draw_y, opacity=0.3)

    def update_position(self, pos):
        if self.is_held:
            self.x, self.y = pos[0], pos[1]

    def rotate_piece(self):
        if not self.is_solved and self.is_held:
            self.angle = (self.angle + 90) % 360

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

# --- 6. DEFINICIÓN DE NIVELES ---
PUZZLE_DEFINITIONS = {
    "gato": {
        "image_file": "images/full_puzzles/pato.jpg", 
        "grid_size": (3, 2), 
        "difficulty": 2
    },
    "ua": {
        "image_file": "images/full_puzzles/ua.jpg", 
        "grid_size": (3, 3), 
        "difficulty": 3
    },
    "alicante": {
        "image_file": "images/full_puzzles/alicante.png", 
        "grid_size": (4, 4), 
        "difficulty": 4 
    },
    "playa": {
        "image_file": "images/full_puzzles/playa.jpg", 
        "grid_size": (4, 4), 
        "difficulty": 4
    },
    "parra": {
        "image_file": "images/full_puzzles/parra.jpg", 
        "grid_size": (5, 5), 
        "difficulty": 5
    }
}

# --- 7. FUNCIÓN: CARGAR PUZZLE ---
def load_puzzle(puzzle_name, grid_origin):
    if puzzle_name not in PUZZLE_DEFINITIONS:
        raise ValueError(f"Puzzle '{puzzle_name}' no definido.")
    
    definition = PUZZLE_DEFINITIONS[puzzle_name]
    rows, cols = definition["grid_size"]
    (ox, oy) = grid_origin
    
    full_image = cv2.imread(definition["image_file"], cv2.IMREAD_UNCHANGED)
    
    if full_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {definition['image_file']}")
        
    if full_image.shape[2] == 3:
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2BGRA)
    elif full_image.shape[2] != 4:
        raise ValueError("La imagen del puzzle debe ser BGR (3 canales) o BGRA (4 canales)")

    img_h, img_w = full_image.shape[:2]
    slice_height = img_h // rows
    slice_width = img_w // cols
    
    pieces = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * slice_height, (r + 1) * slice_height
            x1, x2 = c * slice_width, (c + 1) * slice_width
            
            image_slice = full_image[y1:y2, x1:x2]
            
            target_x = ox + c * PIECE_SIZE
            target_y = oy + r * PIECE_SIZE
            target_angle = 0 
            initial_angle = random.choice([0, 90, 180, 270])
            
            piece = PuzzlePiece(image_slice, target_x, target_y, target_angle, initial_angle)
            pieces.append(piece)
            
    return pieces, definition["grid_size"], definition["difficulty"], full_image

# --- FUNCIÓN DE AUDIO ---
def play_music(file_path, loops=-1):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops) 
        print(f"Reproduciendo música: {file_path}")
    except pygame.error as e:
        print(f"Error al cargar o reproducir música: {file_path}")
        print(f"Detalle del error: {e}")

# --- FUNCIÓN DE MENÚ ---
def show_menu_and_get_selection(landmarker, cap, w, h):
    menu_options_data = [
        ("1", "Facil (3x2): Gato", ["gato"], h//4 + 120),
        ("2", "Normal (3x3): UA", ["ua"], h//4 + 190),
        ("3", "Dificil (4x4): Alicante/Playa", ["alicante", "playa"], h//4 + 260),
        ("4", "Experto (5x5): Parra", ["parra"], h//4 + 330)
    ]
    
    CLICK_WIDTH = 600
    CLICK_X = w//2 - 300 

    selection_areas = []
    for _, _, _, y_text in menu_options_data:
        y1 = y_text - 40 
        y2 = y_text + 15 
        selection_areas.append((CLICK_X, y1, CLICK_X + CLICK_WIDTH, y2))
    
    play_music(MENU_MUSIC_PATH)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: sys.exit(1)
        
        menu_canvas = np.full((h, w, 3), (25, 25, 112), dtype="uint8") 
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        cursor_pos_x, cursor_pos_y = None, None
        
        if result.hand_landmarks:
            for hand_landmarks_list in result.hand_landmarks:
                gesture, pos = detect_gesture(hand_landmarks_list)
                
                cursor_pos_x = int((1.0 - pos[0]) * w) 
                cursor_pos_y = int(pos[1] * h)

                if gesture == "PINCH":
                    cv2.circle(menu_canvas, (cursor_pos_x, cursor_pos_y), 20, (0, 255, 0), -1)
                    
                    for i, area in enumerate(selection_areas):
                        x1, y1, x2, y2 = area
                        
                        if x1 <= cursor_pos_x <= x2 and y1 <= cursor_pos_y <= y2:
                            selected_puzzles = menu_options_data[i][2]
                            return random.choice(selected_puzzles)
                            
                else: 
                    cv2.circle(menu_canvas, (cursor_pos_x, cursor_pos_y), 15, (255, 255, 255), -1)

        cv2.putText(menu_canvas, "SELECCIONA LA DIFICULTAD", (w//2 - 400, h//4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(menu_canvas, "-----------------------------", (w//2 - 400, h//4 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for i, (key, desc, _, y_start) in enumerate(menu_options_data):
            x1, y1, x2, y2 = selection_areas[i]
            color = (255, 255, 255)
            
            if cursor_pos_x is not None and x1 <= cursor_pos_x <= x2 and y1 <= cursor_pos_y <= y2:
                color = (255, 255, 0) 
                cv2.rectangle(menu_canvas, (x1, y1), (x2, y2), (255, 255, 0), cv2.FILLED)

            cv2.putText(menu_canvas, f"Opcion {key}: {desc}", (CLICK_X, y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Juego de Puzzle IPM", menu_canvas) 
        
        if cv2.waitKey(1) & 0xFF == 27: 
            sys.exit()

# --- 8. OPCIONES DE MEDIAPIPE ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# --- 9. BUCLE PRINCIPAL ---
pygame.mixer.init()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    final_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución de la ventana: {final_w}x{final_h}")
    
    rotate_cooldown = 0
    COOLDOWN_FRAMES = 10 
    fist_cooldown = 0 
    FIST_COOLDOWN_FRAMES = 15

    ret, frame = cap.read()
    if not ret: sys.exit(1)
    h, w, _ = frame.shape
    
    WINDOW_NAME = "Juego de Puzzle IPM"
    
    loaded_puzzle_name = show_menu_and_get_selection(landmarker, cap, w, h)
    
    pygame.mixer.music.stop()
    play_music(GAME_MUSIC_PATH)
    
    max_grid_cols = 5 
    max_grid_rows = 5
    grid_w_pixels = max_grid_cols * PIECE_SIZE
    grid_h_pixels = max_grid_rows * PIECE_SIZE
    grid_x = (w - grid_w_pixels) // 2 
    grid_y = (h - grid_h_pixels) // 2 
    GRID_ORIGIN = (grid_x, grid_y)
    HELP_POSITION = (w - HELP_SIZE - 10, h - HELP_SIZE - 50)
    
    print(f"--- Cargando puzzle seleccionado: {loaded_puzzle_name} ---")
    
    try:
        puzzle_pieces, grid_size, difficulty, original_puzzle_img = load_puzzle(loaded_puzzle_name, GRID_ORIGIN)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de haber creado la carpeta 'images/full_puzzles/' y puesto las imágenes allí.")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado al cargar el puzzle: {e}")
        sys.exit(1)
    
    show_help_image = False
    if original_puzzle_img is not None:
        help_image_resized = cv2.resize(original_puzzle_img, (HELP_SIZE, HELP_SIZE), interpolation=cv2.INTER_AREA)
        
        # --- NUEVO: Crear la imagen fantasma completa del puzzle ---
        full_ghost_image = cv2.resize(original_puzzle_img, (grid_size[1] * PIECE_SIZE, grid_size[0] * PIECE_SIZE))
        if full_ghost_image.shape[2] == 3: # Asegurarse de que tenga canal alfa
            full_ghost_image = cv2.cvtColor(full_ghost_image, cv2.COLOR_BGR2BGRA)
            
    else:
        help_image_resized = None
        full_ghost_image = None

    held_piece = None
    game_won = False
    start_time = time.time()
    elapsed_time = 0
    final_score = 0
    
    total_pieces = len(puzzle_pieces)
    # --- RECOMPENSAS (MODIFICADO) ---
    rewards_claimed = {"25": False, "50": False} # Solo 2 recompensas
    banked_hints = 0 # Solo para la del 50%
    active_hint_piece = None 
    show_full_ghost_timer = 0.0 # Temporizador para la pista del 25%

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        canvas = np.full((h, w, 3), (255, 255, 255), dtype="uint8") 

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        current_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, current_timestamp_ms)
        
        if rotate_cooldown > 0: rotate_cooldown -= 1
        if fist_cooldown > 0: fist_cooldown -= 1 

        pinch_pos, point_pos, fist_detected = None, None, False

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
                elif gesture == "FIST": 
                    fist_detected = True
                    cv2.circle(canvas, (cursor_pos_x, cursor_pos_y), 20, (255, 0, 255), -1) 
                elif gesture == "HAND":
                    cv2.circle(canvas, (cursor_pos_x, cursor_pos_y), 10, (0, 0, 255), -1)
        
        if not game_won:
            if held_piece is None:
                if pinch_pos:
                    pinch_pixel_x, pinch_pixel_y = int((1.0 - pinch_pos[0]) * w), int(pinch_pos[1] * h)
                    for piece in reversed(puzzle_pieces):
                        if piece.check_collision((pinch_pixel_x, pinch_pixel_y)): 
                            if piece.is_solved:
                                piece.is_solved = False 
                                if piece == active_hint_piece: 
                                    active_hint_piece = None
                            held_piece = piece
                            held_piece.is_held = True
                            break 
            else: 
                if not pinch_pos:
                    held_piece.is_held = False
                    if held_piece.snap_to_target():
                        if held_piece == active_hint_piece:
                            active_hint_piece = None
                    held_piece = None
                else:
                    pinch_pixel_x, pinch_pixel_y = int((1.0 - pinch_pos[0]) * w), int(pinch_pos[1] * h)
                    held_piece.update_position((pinch_pixel_x, pinch_pixel_y))
                    
                    if point_pos and rotate_cooldown == 0:
                        point_pixel_x, point_pixel_y = int((1.0 - point_pos[0]) * w), int(point_pos[1] * h)
                        if held_piece.check_collision((point_pixel_x, point_pixel_y)):
                            held_piece.rotate_piece()
                            rotate_cooldown = COOLDOWN_FRAMES 
            
            if fist_detected and fist_cooldown == 0 and banked_hints > 0 and active_hint_piece is None:
                banked_hints -= 1
                fist_cooldown = FIST_COOLDOWN_FRAMES
                
                unsolved_pieces = [p for p in puzzle_pieces if not p.is_solved]
                if unsolved_pieces:
                    active_hint_piece = random.choice(unsolved_pieces)
                    print(f"¡PISTA (50%) ACTIVADA! Mostrando pieza fantasma.")

        # --- DIBUJAR CUADRÍCULA Y PIEZAS ---
        draw_grid(canvas, GRID_ORIGIN, grid_size, PIECE_SIZE)
        
        # --- DIBUJAR PISTA 25% (Pista completa) ---
        if time.time() < show_full_ghost_timer:
            if full_ghost_image is not None:
                (ox, oy) = GRID_ORIGIN
                overlay_transparent(canvas, full_ghost_image, ox, oy, opacity=0.3)
        
        # --- DIBUJAR PISTA 50% (Pieza única) ---
        if active_hint_piece:
            active_hint_piece.draw_ghost(canvas)
            
        for piece in puzzle_pieces:
            if piece is not held_piece:
                piece.draw(canvas)
        if held_piece:
            held_piece.draw(canvas)

        current_elapsed = time.time() - start_time 
        
        # --- LÓGICA DE RECOMPENSAS (MODIFICADA) ---
        if not game_won:
            solved_count = sum(1 for p in puzzle_pieces if p.is_solved)
            progress_percent = (solved_count / total_pieces) * 100

            # --- Checkpoint 25% (Pista completa automática) ---
            if progress_percent >= 25 and not rewards_claimed["25"]:
                rewards_claimed["25"] = True
                show_full_ghost_timer = time.time() + FULL_GHOST_DURATION_SECONDS
                print(f"¡RECOMPENSA 25%! Mostrando pista completa por {FULL_GHOST_DURATION_SECONDS} segundos.")
            
            # --- Checkpoint 50% (Pista de pieza) ---
            if progress_percent >= 50 and not rewards_claimed["50"]:
                rewards_claimed["50"] = True
                banked_hints += 1
                print("¡RECOMPENSA 50%! +1 Pista de Pieza ganada (Usar Puño).")
                        
            # (Recompensa 75% eliminada)

            # Ayuda por tiempo (la mantenemos)
            if current_elapsed >= HELP_DELAY_SECONDS and not show_help_image:
                show_help_image = True
                print("¡AYUDA ACTIVADA! Se muestra la imagen completa del puzzle.")

            if all(p.is_solved for p in puzzle_pieces):
                game_won = True
                end_time = time.time()
                elapsed_time = end_time - start_time
                final_score = (10000 * difficulty) / elapsed_time 
                print(f"¡HAS GANADO! Tiempo: {elapsed_time:.2f}s, Puntuación: {final_score:.0f}")
                pygame.mixer.music.stop()
                play_music(VICTORY_MUSIC_PATH,1)
                
        if show_help_image and help_image_resized is not None:
            (hx, hy) = HELP_POSITION
            overlay_transparent(canvas, help_image_resized, hx, hy)

        # --- DIBUJAR BARRA DE PROGRESO (MODIFICADA) ---
        bar_x, bar_y, bar_w, bar_h = w - 320, 10, 300, 30
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), 2)
        fill_w = int((progress_percent / 100) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
        
        # Check marks (solo 25 y 50)
        for perc in [25, 50]:
            cx = bar_x + int((perc / 100) * bar_w)
            color = (0, 255, 0) if rewards_claimed[str(perc)] else (0, 0, 0)
            cv2.line(canvas, (cx, bar_y), (cx, bar_y + bar_h), color, 2)
        
        cv2.putText(canvas, f"Pistas: {banked_hints}", (bar_x, bar_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # --- MOSTRAR PANTALLA DE VICTORIA O TIEMPO ---
        if game_won:
            cv2.putText(canvas, "HAS GANADO", (w//2 - 190, h//2 - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            cv2.putText(canvas, f"Tiempo: {elapsed_time:.2f}s", (w//2 - 150, h//2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(canvas, f"Puntuacion: {final_score:.0f}", (w//2 - 160, h//2 + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        else:
            cv2.putText(canvas, f"Tiempo: {current_elapsed:.1f}", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
            
    pygame.mixer.music.stop()
    cap.release()
    cv2.destroyAllWindows()