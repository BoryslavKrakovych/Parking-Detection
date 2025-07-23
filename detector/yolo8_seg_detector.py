import cv2  # OpenCV для обробки зображень
import numpy as np  # для роботи з масивами та масками
import torch  # PyTorch — для обчислень на GPU/CPU
from ultralytics import YOLO  # бібліотека Ultralytics з моделями YOLOv8


class YOLOv8SegDetector:
    def __init__(self, model_path="weights/yolov8n-seg.pt", conf=0.25, device='cpu'):
        """
        Ініціалізація YOLOv8 сегментаційної моделі.

        параметри:
            model_path (str): шлях до ваг моделі (файл .pt)
            conf (float): поріг впевненості для фільтрації результатів
            device (str): обчислювальний пристрій ('cpu' або 'cuda')
        """
        self.model = YOLO(model_path)  # завантаження моделі з диску
        self.model.to(device)  # переносимо модель на обране залізо
        self.conf = conf  # зберігаємо поріг впевненості

    def segment(self, frame):
        """
        Застосовує модель YOLOv8 для сегментації об'єктів на кадрі.

        повертає тільки маски та координати об'єктів класів car, truck, bus.

        параметри:
            frame (np.ndarray): кадр з відео (зображення BGR)

        повертає:
            masks (list[np.ndarray]): список сегментованих масок (float32)
            classes (list[int]): список класів для кожної маски
            boxes (list[list[int]]): координати bbox [x1, y1, x2, y2]
        """
        results = self.model(frame)[0]  # виконання інференсу; беремо перший результат з batch

        masks = []  # маски об'єктів
        classes = []  # їхні класи
        boxes = []  # відповідні bounding boxes

        # перевірка, чи взагалі модель згенерувала маски
        if results.masks is not None:
            for i, m in enumerate(results.masks.data):
                cls = int(results.boxes.cls[i])  # отримуємо клас об'єкта

                # залишаємо лише транспортні засоби (автомобілі, автобуси тощо)
                if cls in [1, 2, 3, 5, 7]:  # bicycle, car, motorcycle, bus, truck
                    raw_mask = m.cpu().numpy()  # переводимо в NumPy-масив без бінаризації

                    # координати bbox (xyxy формат)
                    xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()

                    # маска завжди має бути масштабована до розміру кадру
                    full_mask = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]))

                    # додаємо в результати
                    masks.append(full_mask)
                    classes.append(cls)
                    boxes.append([x1, y1, x2, y2])

        return masks, classes, boxes  # повертаємо всі дані для подальшої обробки
