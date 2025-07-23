import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np

def draw_roi(frame, roi, status="free"):
    """
    Малює область ROI на кадрі з відповідним кольором заливки, що відображає її статус.

    Args:
        frame (np.ndarray): кадр відео, на який наноситься розмітка.
        roi (dict): словник з ключем "points", що містить координати вершини полігону.
        status (str): статус місця — "free", "occupied" або "bad".
                      Від нього залежить колір заливки:
                      - "free" → зелений
                      - інші → червоний
    """
    # Перетворення координат у формат OpenCV
    points = np.array(roi['points'], np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()

    # Вибір кольору заповнення за статусом
    if status == "free":
        fill_color = (0, 255, 0)     # зелений
    else:
        fill_color = (0, 0, 255)     # червоний

    # Накладення півпрозорої заливки на ROI
    alpha = 0.4
    cv2.fillPoly(overlay, [points], color=fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Обводка ROI білим контуром
    cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 255), thickness=1)

#
# def draw_bad_parking_mask(frame, mask):
#     """
#     Малює контур неправильно припаркованого автомобіля (який порушує межі ROI).
#
#     Args:
#         frame (np.ndarray): кадр відео.
#         mask (np.ndarray): бінарна маска (0/1) автомобіля.
#     """
#     # Масштабування маски до uint8 (0 або 255)
#     mask_uint8 = (mask * 255).astype(np.uint8)
#
#     # Пошук контурів на масці
#     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Малювання контурів жовтим кольором
#     cv2.drawContours(frame, contours, -1, (0, 255, 255), thickness=2)  # жовтий

def draw_segmentation_mask(frame, mask):
    """
    Малює контур сегментованої маски авто жовтим кольором.

    Args:
        frame (np.ndarray): оригінальний кадр відео.
        mask (np.ndarray): бінарна маска авто (0 або 1).
    """
    # Масштабування маски до розміру кадру
    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Перетворення маски у формат для пошуку контурів (0 або 255)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)

    # Знаходимо контури об'єкта
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Малюємо знайдені контури жовтим кольором
    cv2.drawContours(frame, contours, -1, color=(0, 255, 255), thickness=2)  # жовтий