import sys
sys.dont_write_bytecode = True  # Не створювати .pyc-файли
import cv2
import json
import os
import numpy as np

# === Конфігурації ===
CONFIG_PATH = "config/parking_config.json"  # Шлях до JSON з ROI
WINDOW_NAME = "Add ROI"                     # Назва вікна OpenCV
TARGET_SIZE = (1152, 640)                   # Розмір зображення (w, h)
HEADER_HEIGHT = 100                         # Висота панелі інструкцій
FOOTER_HEIGHT = 40                          # Висота нижньої панелі

# === Завантаження першого кадру з відео ===
cap = cv2.VideoCapture("video.mp4")
ret, original_frame = cap.read()
cap.release()

if not ret:
    print("Error loading video.")
    exit()

# Отримання розміру оригінального кадру
orig_h, orig_w = original_frame.shape[:2]
scale_w, scale_h = TARGET_SIZE[0] / orig_w, TARGET_SIZE[1] / orig_h

# Масштабування кадру до TARGET_SIZE
base_frame = cv2.resize(original_frame, TARGET_SIZE)

# === Завантаження вже існуючих ROI (якщо є) ===
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        roi_list = json.load(f)
else:
    roi_list = []

# Початкові змінні для розмітки
next_id = max([roi["id"] for roi in roi_list], default=-1) + 1
current_points = []  # Поточні 4 точки полігону
show_ids = False     # Показувати ID полігонів чи ні

# === Обробка кліків миші ===
def mouse_callback(event, x, y, flags, param):
    global current_points, next_id, roi_list

    if event == cv2.EVENT_LBUTTONDOWN:
        if y < HEADER_HEIGHT:
            return  # Клік у панель інструкцій — ігноруємо
        y_adj = y - HEADER_HEIGHT  # Врахування зсуву вниз
        current_points.append((x, y_adj))
        if len(current_points) == 4:
            # Нормалізуємо координати до [0..1]
            norm_points = [
                (round(px / TARGET_SIZE[0], 6), round(py / TARGET_SIZE[1], 6))
                for px, py in current_points
            ]
            roi_list.append({
                "id": next_id,
                "points": norm_points
            })
            print(f"[+] ROI added: ID = {next_id}")
            next_id += 1
            current_points.clear()

    elif event == cv2.EVENT_MBUTTONDOWN:
        if y < HEADER_HEIGHT:
            return
        y_adj = y - HEADER_HEIGHT
        for roi in roi_list:
            scaled_points = [
                (int(px * TARGET_SIZE[0]), int(py * TARGET_SIZE[1]))
                for px, py in roi["points"]
            ]
            contour = np.array(scaled_points, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y_adj), False) >= 0:
                print(f"[-] ROI deleted: ID = {roi['id']}")
                roi_list = [r for r in roi_list if r["id"] != roi["id"]]
                break

# === Візуалізація кадру з ROI та інструкціями ===
def draw(frame, show_ids):
    h, w = frame.shape[:2]

    # Панель інструкцій (жовта смуга зверху)
    header = np.full((HEADER_HEIGHT, w, 3), (0, 255, 255), dtype=np.uint8)
    footer = np.full((FOOTER_HEIGHT, w, 3), (0, 255, 255), dtype=np.uint8)

    # Текстові інструкції
    instructions = [
        "MODE: ADD ROI",
        "LMB - Click 4 Points to add ROI",
        "MMB - Delete ROI by clicking inside",
        "S - Save to JSON file",
        "Q - Quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(header, text, (10, 25 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Малювання полігонів ROI
    for roi in roi_list:
        scaled_points = [(int(px * w), int(py * h)) for px, py in roi["points"]]
        pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

        # Виведення ID у центрі полігону
        if show_ids:
            cx = int(np.mean([pt[0] for pt in scaled_points]))
            cy = int(np.mean([pt[1] for pt in scaled_points]))
            cv2.putText(frame, f"{roi['id']}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Панель перемикання індексів
    toggle_text = f"Index: {'ON' if show_ids else 'OFF'} (Press 'i' to {'hide' if show_ids else 'show'})"
    cv2.putText(footer, toggle_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return np.vstack((header, frame, footer))

# === Налаштування вікна OpenCV ===
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.resizeWindow(WINDOW_NAME, TARGET_SIZE[0], TARGET_SIZE[1] + HEADER_HEIGHT + FOOTER_HEIGHT)

print("""
ROI Drawing Instructions:
-------------------------------------
LMB: Click 4 times to define polygon (ROI)
MMB: Click inside ROI to delete
Press 's': Save to JSON
Press 'q': Quit
""")

# === Основний цикл розмітки ===
while True:
    temp = base_frame.copy()
    temp = draw(temp, show_ids)
    cv2.imshow(WINDOW_NAME, temp)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        with open(CONFIG_PATH, "w") as f:
            json.dump(roi_list, f, indent=2)
        print("[Saved ROI to config.]")
    elif key == ord('q'):
        break
    elif key == ord('i'):
        show_ids = not show_ids
        print(f"[TOGGLE] Show Index IDs: {'ON' if show_ids else 'OFF'}")

cv2.destroyAllWindows()