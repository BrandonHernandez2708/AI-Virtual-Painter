import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
import urllib.request

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode


# ═══════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajusta estos valores para tus pruebas
# ═══════════════════════════════════════════════════════

# Resolución de captura (afecta rendimiento y precisión)
# Opciones comunes: 320x240 | 640x480 | 1280x720
CAPTURE_W = 1080
CAPTURE_H = 720

# Resolución de visualización (solo escala el display, sin costo de procesamiento)
# Puede ser mayor o menor que la captura
# Opciones comunes: 640x480 | 800x600 | 1280x720 | 1920x1080
DISPLAY_W = 1280
DISPLAY_H = 720

# Pantalla completa: True ignora DISPLAY_W/DISPLAY_H y ocupa toda la pantalla
FULLSCREEN = False

# FPS objetivo del loop principal
TARGET_FPS = 20

# ═══════════════════════════════════════════════════════

MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

if not os.path.exists(MODEL_PATH):
    print("[INFO] Descargando modelo MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Modelo descargado.")


# ─────────────────────────────────────────────
#  Detección de cámaras disponibles
# ─────────────────────────────────────────────
def scan_cameras(max_index: int = 6) -> list[dict]:
    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({'index': i, 'name': f'Camara {i}', 'width': w, 'height': h})
            cap.release()
    return cameras


# ─────────────────────────────────────────────
#  UI del selector
# ─────────────────────────────────────────────
def draw_selector_ui(frame: np.ndarray, cameras: list[dict], current: int) -> np.ndarray:
    h, w    = frame.shape[:2]
    panel_w = 270

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, h), (12, 12, 18), -1)
    cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "SELECCIONAR CAMARA", (14, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 210), 1)
    cv2.line(frame, (14, 40), (panel_w - 14, 40), (55, 55, 65), 1)

    item_h  = 58
    start_y = 52

    for idx, cam in enumerate(cameras):
        y0       = start_y + idx * item_h
        y1       = y0 + item_h - 4
        selected = idx == current

        bg = (28, 110, 82) if selected else (22, 22, 30)
        cv2.rectangle(frame, (10, y0), (panel_w - 10, y1), bg, -1)
        if selected:
            cv2.rectangle(frame, (10, y0), (14, y1), (0, 210, 140), -1)

        txt_col = (245, 245, 245) if selected else (170, 170, 185)
        cv2.putText(frame, cam['name'], (22, y0 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, txt_col, 1)
        cv2.putText(frame, f"{cam['width']}x{cam['height']}", (22, y0 + 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (80, 170, 130), 1)

    foot_y = start_y + len(cameras) * item_h + 20
    for line in ["W/S o Flechas: navegar", "ENTER/SPACE: confirmar", "Q: salir"]:
        cv2.putText(frame, line, (12, foot_y),
                    cv2.FONT_HERSHEY_PLAIN, 0.88, (80, 80, 95), 1)
        foot_y += 16

    return frame


def select_camera(cameras: list[dict]) -> int | None:
    if not cameras:
        print("[ERROR] Sin camaras disponibles.")
        return None
    if len(cameras) == 1:
        print(f"[INFO] Una sola camara detectada (indice {cameras[0]['index']}).")
        return cameras[0]['index']

    current = 0
    preview = cv2.VideoCapture(cameras[current]['index'], cv2.CAP_DSHOW)
    WIN     = 'Selector de Camara'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)
    chosen  = None

    while True:
        ret, frame = preview.read()
        if not ret or frame is None:
            frame = np.zeros((480, 800, 3), np.uint8)
        frame = cv2.resize(frame, (800, 480))
        frame = draw_selector_ui(frame, cameras, current)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key in (82, ord('w'), ord('W')):
            new = (current - 1) % len(cameras)
            if new != current:
                preview.release()
                current = new
                preview = cv2.VideoCapture(cameras[current]['index'], cv2.CAP_DSHOW)
        elif key in (84, ord('s'), ord('S')):
            new = (current + 1) % len(cameras)
            if new != current:
                preview.release()
                current = new
                preview = cv2.VideoCapture(cameras[current]['index'], cv2.CAP_DSHOW)
        elif key in (13, 32):
            chosen = cameras[current]['index']
            break

    preview.release()
    cv2.destroyWindow(WIN)
    return chosen


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
def load_header_images(folder: str, target_width: int) -> list[np.ndarray]:
    images = []
    if not os.path.isdir(folder):
        return images
    for name in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, name))
        if img is not None:
            images.append(cv2.resize(img, (target_width, 125)))
    return images


def make_fallback_header(target_width: int) -> list[np.ndarray]:
    palette = [(0,0,255), (255,0,0), (0,255,0), (0,0,0)]
    labels  = ['Rojo', 'Azul', 'Verde', 'Borrar']
    headers = []
    step    = target_width // 4
    for sel in range(4):
        hdr = np.full((125, target_width, 3), 30, np.uint8)
        for i, (c, lbl) in enumerate(zip(palette, labels)):
            active = (i == sel)
            cv2.rectangle(hdr, (i*step+4, 8), ((i+1)*step-4, 117),
                          c if active else tuple(x//3 for x in c), -1)
            cv2.putText(hdr, lbl, (i*step + 10, 72),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
            if active:
                cv2.rectangle(hdr, (i*step+4, 8), ((i+1)*step-4, 117), (255,255,255), 2)
        headers.append(hdr)
    return headers


# ─────────────────────────────────────────────
#  Bucle principal
# ─────────────────────────────────────────────
def run(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {camera_index}.")
        return

    # Resolución real que entregó la cámara (puede diferir de la solicitada)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Captura: {real_w}x{real_h}  |  Display: ", end="")
    print("Pantalla completa" if FULLSCREEN else f"{DISPLAY_W}x{DISPLAY_H}")

    folder_path  = os.path.join(os.path.dirname(__file__), 'Header')
    overlay_list = load_header_images(folder_path, real_w)
    if not overlay_list:
        print("[WARN] Carpeta Header no encontrada. Usando header generado.")
        overlay_list = make_fallback_header(real_w)

    palette    = [(0,0,255), (255,0,0), (0,255,0), (0,0,0)]
    header     = overlay_list[0]
    draw_color = palette[0]
    thickness  = 20
    tip_ids    = [4, 8, 12, 16, 20]
    xp, yp     = 0, 0
    img_canvas = np.zeros((real_h, real_w, 3), np.uint8)

    hand_options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Configurar ventana de display
    WIN = 'MediaPipe Hands'
    if FULLSCREEN:
        cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)

    frame_time = 1.0 / TARGET_FPS
    last_time  = time.time()

    with hand_landmarker.HandLandmarker.create_from_options(hand_options) as hands:
        while cap.isOpened():
            now = time.time()
            if (now - last_time) < frame_time:
                time.sleep(frame_time - (now - last_time))
            last_time = time.time()

            cap.grab(); cap.grab()
            success, image = cap.read()
            if not success:
                break

            image    = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results  = hands.detect_for_video(mp_image, int(time.time() * 1000))
            image    = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    points = [[int(lm.x * real_w), int(lm.y * real_h)]
                              for lm in hand_landmarks]
                    if not points:
                        continue

                    x1, y1 = points[8]
                    x2, y2 = points[12]
                    x3, y3 = points[4]
                    x4, y4 = points[20]

                    fingers = [1 if points[tip_ids[0]][0] < points[tip_ids[0]-1][0] else 0]
                    for id in range(1, 5):
                        fingers.append(1 if points[tip_ids[id]][1] < points[tip_ids[id]-2][1] else 0)

                    # Selección de color
                    if (fingers[1] and fingers[2]) and all(fingers[i]==0 for i in [0,3,4]):
                        xp, yp = x1, y1
                        if y1 < 125:
                            zone = min(x1 // (real_w // 4), len(overlay_list) - 1)
                            header     = overlay_list[zone]
                            draw_color = palette[zone]
                        cv2.rectangle(image, (x1-10,y1-15), (x2+10,y2+23), draw_color, cv2.FILLED)

                    # Stand By
                    if (fingers[1] and fingers[4]) and all(fingers[i]==0 for i in [0,2,3]):
                        cv2.line(image, (xp,yp), (x4,y4), draw_color, 5)
                        xp, yp = x1, y1

                    # Dibujo
                    if fingers[1] and all(fingers[i]==0 for i in [0,2,3,4]):
                        cv2.circle(image, (x1,y1), int(thickness/2), draw_color, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(img_canvas, (xp,yp), (x1,y1), draw_color, thickness)
                        xp, yp = x1, y1

                    # Limpiar
                    if all(fingers[i]==0 for i in range(5)):
                        img_canvas = np.zeros((real_h, real_w, 3), np.uint8)
                        xp, yp     = x1, y1

                    # Grosor
                    if (all(fingers[i]==j for i,j in zip(range(5),[1,1,0,0,0])) or
                            all(fingers[i]==j for i,j in zip(range(5),[1,1,0,0,1]))):
                        r      = int(math.sqrt((x1-x3)**2+(y1-y3)**2)/3)
                        x0, y0 = (x1+x3)/2, (y1+y3)/2
                        v1, v2 = -(y1-y3), (x1-x3)
                        mod_v  = math.sqrt(v1**2+v2**2)
                        if mod_v > 0:
                            v1, v2 = v1/mod_v, v2/mod_v
                        c      = 3 + r
                        x0, y0 = int(x0-v1*c), int(y0-v2*c)
                        cv2.circle(image, (x0,y0), int(r/2), draw_color, -1)
                        if fingers[4]:
                            thickness = r
                            cv2.putText(image, 'Check', (x4-25,y4-8),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)
                        xp, yp = x1, y1

            # Composición canvas + cámara
            image[0:125, 0:real_w] = header
            img_gray   = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
            img_inv    = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            img        = cv2.bitwise_and(image, img_inv)
            img        = cv2.bitwise_or(img, img_canvas)

            # Escalar al tamaño de display (solo si no es fullscreen)
            if not FULLSCREEN:
                img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))

            cv2.imshow(WIN, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("[INFO] Escaneando camaras...")
    cameras = scan_cameras(max_index=6)

    if not cameras:
        print("[ERROR] No se detecto ninguna camara.")
    else:
        print(f"[INFO] {len(cameras)} camara(s) encontrada(s): {[c['index'] for c in cameras]}")
        chosen = select_camera(cameras)
        if chosen is not None:
            print(f"[INFO] Iniciando con camara {chosen}...")
            run(chosen)
        else:
            print("[INFO] Seleccion cancelada.")