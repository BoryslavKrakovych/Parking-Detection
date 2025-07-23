import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import cv2
import sys
import os

# Додаємо шлях до локального репозиторію YOLOv5
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))  # Коренева директорія проєкту
YOLOV5_PATH = os.path.join(ROOT, "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Імпорт необхідних модулів з YOLOv5
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_segments
from yolov5.utils.augmentations import letterbox
from yolov5.utils.segment.general import process_mask


class YOLOSegDetector:
    """
    Клас для сегментації об'єктів за допомогою моделі YOLOv5n-seg.
    """

    def __init__(self, model_path="weights/yolov5n-seg.pt", conf=0.25, imgsz=1152, device='cpu'):
        """
        Ініціалізація сегментаційного детектора.

        :param model_path: шлях до моделі .pt
        :param conf: поріг впевненості для фільтрації слабких об'єктів
        :param imgsz: розмір зображення для вхідного інференсу
        :param device: пристрій для обчислень ('cpu' або 'cuda')
        """
        self.device = device
        self.model = DetectMultiBackend(model_path, device=torch.device(device))  # Завантаження моделі
        self.model.eval()  # Режим оцінки (без градієнтів)
        self.conf = conf
        self.imgsz = imgsz

    def segment(self, frame):
        """
        Сегментація кадру без letterbox (використовується вже підготовлений frame).
        """
        img0 = frame.copy()
        h, w = img0.shape[:2]

        # Перетворення BGR → RGB → CHW → Tensor
        img = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            out = self.model(img_tensor)
            pred = out[0]
            proto = out[1] if len(out) > 1 else None

        det = non_max_suppression(pred, self.conf, self.conf, classes=None, agnostic=False, max_det=1000, nm=32)[0]

        masks = []
        classes = []

        if det is not None and len(det) and proto is not None:
            masks_pred = process_mask(proto[0], det[:, 6:], det[:, :4], img_tensor.shape[2:], upsample=True)
            masks_pred = masks_pred.cpu().numpy()
            det_classes = det[:, 5].int().cpu().tolist()

            for i, mask in enumerate(masks_pred):
                mask_bin = (mask > 0.5).astype(np.uint8)
                masks.append(mask_bin)
                classes.append(det_classes[i])

        return masks, classes
