# --- Імпорти стандартних бібліотек ---
import requests
import io
from PIL import Image
import cv2  # OpenCV: для обробки зображень та відео
import json  # Для читання JSON-конфігурацій (наприклад, ROI)
import time  # Для вимірювання часу обробки кадру (FPS тощо)
import os  # Для перевірки існування файлів та запуску скриптів
import numpy as np  # Масиви, математика, координати
import tkinter as tk  # GUI: для вибору джерела відео
from tkinter import simpledialog  # Ввід тексту через діалогове вікно

# --- Імпорти проєктних модулів ---
from detector.yolo8_seg_detector import YOLOv8SegDetector  # Клас сегментаційного YOLOv8
from local_utils.track import Tracker  # Трекер з Kalman-фільтром
from local_utils.drawing import draw_roi, draw_segmentation_mask  # Візуалізація ROI та масок

SEND_INTERVAL = 1  # кожні 30 кадрів (приблизно ~1 сек при 30 FPS)
server_url_frame = "http://127.0.0.1:8080/upload_frame"
server_url_status = "http://127.0.0.1:8080/upload_status"
frame_counter = 0
last_sent_status = None
# --- Конвертація нормалізованих ROI у пікселі ---
def convert_normalized_rois(normalized_rois, width, height):
    """
    Перетворює ROI, де координати [0, 1], у координати зображення (пікселі)
    """
    converted = []
    for roi in normalized_rois:
        pts = np.array(roi["points"], dtype=float)
        pixel_pts = (pts * [width, height]).round().astype(int)  # масштабування до пікселів
        new_roi = roi.copy()
        new_roi["points"] = pixel_pts.tolist()
        converted.append(new_roi)
    return converted

# --- Отримання шляху до відео ---
def get_video_source():
    """
    Вікно вибору джерела відео: локальний файл або IP/RTSP потік
    """
    root = tk.Tk()
    root.withdraw()  # приховати головне вікно
    user_input = simpledialog.askstring(
        title="Select Video Source",
        prompt="Enter camera IP/URL (or leave blank for default):"
    )
    if not user_input:
        return "video.mp4"  # дефолтне відео
    elif user_input.startswith(("rtsp://", "http://")) or user_input.endswith(".mp4"):
        return user_input
    else:
        return f"rtsp://{user_input}/live"

# --- Створення маски з полігону (ROI) ---
def polygon_to_mask(polygon, shape):
    """
    Перетворює список точок (ROI) у бінарну маску
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

# --- Основна ініціалізація ---
VIDEO_PATH = get_video_source()  # джерело відео
print(f"[INFO] Using source: {VIDEO_PATH}")

# --- Параметри ---
show_frame = False
ROI_PATH = "config/parking_config.json"  # шлях до ROI

# --- Перевірка існування ROI-конфігу ---
if not os.path.exists(ROI_PATH):
    print(f"[INFO] ROI-файл не знайдено. Запуск roi_adder.py для створення розмітки...")
    os.system("python roi_adder.py")
    if not os.path.exists(ROI_PATH):
        print(f"[ERROR] ROI-файл так і не створено. Завершення.")
        exit(1)

# --- Завантаження ROI та конвертація ---
with open(ROI_PATH) as f:
    n_rois = json.load(f)
rois = convert_normalized_rois(n_rois, 1152, 640)

# --- Ініціалізація об’єктів ---
yolo = YOLOv8SegDetector()  # сегментаційна модель
tracker = Tracker()  # трекер
cap = cv2.VideoCapture(VIDEO_PATH)  # відеозахоплення
show_ids = False  # чи показувати ID ROI

# --- Відеозапис з візуалізацією ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("video_ROI.mp4", fourcc, 25.0, (1152, 640 + 80 + 40))  # +header+footer

# --- Кешовані обмежувальні прямокутники ROI ---
cached_roi_bboxes = None

# --- Основний цикл обробки кадрів ---
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1152, 640))  # масштабування до фіксованого розміру

    # --- Обчислення ROI-прямокутників ---
    if cached_roi_bboxes is None:
        cached_roi_bboxes = []
        for roi in rois:
            pts = np.array(roi["points"], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            cached_roi_bboxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2)
    roi_bboxes = cached_roi_bboxes

    # --- Детекція об’єктів ---
    yolo_start = time.time()
    masks, classes, boxes = yolo.segment(frame)  # сегментація об’єктів
    # for mask in masks:
    #     draw_segmentation_mask(frame, mask)

    tracks = tracker.update(boxes, roi_bboxes)  # оновлення трекера

    yolo_end = time.time()

    # --- Визначення статусів ROI ---
    roi_statuses = tracker.get_roi_statuses(len(rois))  # occupied / free
    occupied_count = sum(1 for v in roi_statuses.values() if v == "occupied")
    free_count = len(rois) - occupied_count

    # --- Візуалізація ROI на кадрі ---
    for i, roi in enumerate(rois):
        draw_roi(frame, roi, roi_statuses[i])
        if show_ids:
            cx = int(np.mean([p[0] for p in roi["points"]]))
            cy = int(np.mean([p[1] for p in roi["points"]]))
            cv2.putText(frame, str(roi["id"]), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Панель header ---
    header = np.full((80, frame.shape[1], 3), (0, 255, 255), dtype=np.uint8)
    text = f"Free: {free_count}     Occupied: {occupied_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(header, text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    # --- Панель footer ---
    footer = np.full((40, frame.shape[1], 3), (0, 255, 255), dtype=np.uint8)
    toggle_text = f"Index: {'ON' if show_ids else 'OFF'} (Press 'i' to {'hide' if show_ids else 'show'}) Press 'q' to exit "
    cv2.putText(footer, toggle_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- Об’єднання панелей та показ ---
    combined = np.vstack((header, frame, footer))
    cv2.imshow("Parking Detection", combined)
    out.write(combined)

    # --- Обрахунок FPS ---
    end_time = time.time()
    yolo_inference_time = (yolo_end - yolo_start) * 1000
    frame_time = (end_time - start_time) * 1000
    fps = 1000 / frame_time if frame_time > 0 else 0
    print(f"[PERF] YOLO inference: {yolo_inference_time:.2f} ms | Frame time: {frame_time:.2f} ms | FPS: {fps:.2f}")
    # --- Відправка кадру та статусів кожні N кадрів ---
    frame_counter += 1
    if frame_counter % SEND_INTERVAL == 0:
        # --- Відправка кадру ---
        try:
            _, buffer = cv2.imencode('.jpg', combined)
            image_bytes = io.BytesIO(buffer)
            response = requests.post(
                server_url_frame,
                files={"frame": ("frame.jpg", image_bytes.getvalue(), "image/jpeg")}
            )
            print(f"[UPLOAD] Кадр: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Відправка кадру: {e}")

        # --- Відправка статусів ---
        try:
            payload = {
                "free": free_count,
                "occupied": occupied_count,
                "rois": roi_statuses
            }
            if payload != last_sent_status:
                response = requests.post(server_url_status, json=payload)
                print(f"[UPLOAD] Статуси: {response.status_code}")
                last_sent_status = payload
        except Exception as e:
            print(f"[ERROR] Відправка статусів: {e}")
            
    # --- Обробка клавіш ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # вихід
        break
    elif key == ord('i'):  # перемикання ID
        show_ids = not show_ids
        print(f"[TOGGLE] Show IDs: {'ON' if show_ids else 'OFF'}")

# --- Завершення ---
cap.release()
cv2.destroyAllWindows()