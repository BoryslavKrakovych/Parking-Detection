import cv2  # OpenCV для Kalman-фільтра
import numpy as np  # робота з координатами та відстанями
from collections import deque  # для збереження історії координат

class Track:
    """
    Клас для одного треку (одного об'єкта), який використовує Kalman-фільтр і перевіряє
    стабільність присутності в ROI на основі масок.
    """
    def __init__(self, track_id, centroid):
        self.id = track_id  # унікальний ID треку

        # ініціалізація Kalman-фільтра з 4 станами та 2 спостереженнями
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre[:2] = np.array(centroid, np.float32).reshape(2, 1)

        self.history = deque(maxlen=15)  # історія центрів
        self.history.append(centroid)

        self.roi_overlap_history = {}  # історія перекриття по ROI
        self.frame_presence = {}  # чи була маска присутня
        self.status_per_roi = {}  # статус кожного ROI: occupied/free

        self.missing_mask_counter = {}  # скільки кадрів підряд маска відсутня
        self.present_counter = {}  # скільки кадрів підряд маска присутня

    def predict(self):
        """ прогноз нового центру об'єкта за допомогою Kalman-фільтра """
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])

    def update(self, centroid):
        """ оновлення Kalman-фільтра новим спостереженням (центром bbox) """
        measurement = np.array(centroid, dtype=np.float32).reshape(2, 1)
        self.kalman.correct(measurement)
        self.history.append(centroid)

    def update_overlap(self, roi_id, overlap):
        """
        оновлення історії перекриття з певним ROI
        зберігає як числове перекриття, так і присутність (>=0.2)
        """
        if roi_id not in self.roi_overlap_history:
            self.roi_overlap_history[roi_id] = deque(maxlen=15)
        if roi_id not in self.frame_presence:
            self.frame_presence[roi_id] = deque(maxlen=15)

        self.roi_overlap_history[roi_id].append(overlap)
        self.frame_presence[roi_id].append(overlap >= 0.2)

        self._update_status(roi_id)

    def _update_status(self, roi_id):
        """
        оновлює статус ROI для треку:
        - якщо перекриває ≥5 кадрів → occupied
        - якщо зникає ≥5 кадрів → free
        """
        pres = self.frame_presence[roi_id]

        # ініціалізація, якщо вперше
        if roi_id not in self.missing_mask_counter:
            self.missing_mask_counter[roi_id] = 0
        if roi_id not in self.present_counter:
            self.present_counter[roi_id] = 0
        if roi_id not in self.status_per_roi:
            self.status_per_roi[roi_id] = "free"

        if pres[-1]:  # останній кадр — маска присутня
            self.present_counter[roi_id] += 1
            self.missing_mask_counter[roi_id] = 0

            if self.present_counter[roi_id] >= 15:
                if self.status_per_roi[roi_id] != "occupied":
                    # додаткова перевірка стабільності
                    if self.is_stable_in_roi(roi_id) and self.is_center_stable():
                        self.status_per_roi[roi_id] = "occupied"

        else:  # останній кадр — маска зникла
            self.missing_mask_counter[roi_id] += 1
            self.present_counter[roi_id] = 0

            if self.missing_mask_counter[roi_id] >= 15:
                if self.status_per_roi[roi_id] != "free":
                    self.status_per_roi[roi_id] = "free"

    def is_stable_in_roi(self, roi_id, min_overlap=0.2):
        """
        перевіряє, чи всі останні N кадрів перекриття з ROI були достатніми
        """
        overlaps = self.roi_overlap_history.get(roi_id, [])
        return len(overlaps) == overlaps.maxlen and all(o >= min_overlap for o in overlaps)

    def is_center_stable(self, threshold=30):
        """
        перевіряє, чи центр об'єкта стабільний (не змістився різко між останніми 2 кадрами)
        """
        if len(self.history) < 2:
            return False
        prev = np.array(self.history[-2])
        curr = np.array(self.history[-1])
        return np.linalg.norm(prev - curr) < threshold


class Tracker:
    """
    головний менеджер треків: оновлює їх, створює нові та визначає статус ROI
    """
    def __init__(self, max_distance=100):
        self.tracks = {}  # активні треки
        self.next_id = 0  # ID для наступного нового треку
        self.max_distance = max_distance  # максимальна дистанція для матчінгу bbox

    def update(self, detections, roi_bboxes=None):
        """
        оновлює треки на основі нових детекцій
        якщо об'єкт близький до існуючого треку — оновлюється
        інакше створюється новий трек
        """
        used_ids = set()
        updated_tracks = {}

        for det in detections:
            x1, y1, x2, y2 = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            matched_id = None
            min_dist = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                last_cx, last_cy = track.history[-1]
                dist = np.hypot(cx - last_cx, cy - last_cy)
                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    matched_id = track_id

            if matched_id is not None:
                # оновлюємо існуючий трек
                track = self.tracks[matched_id]
                track.update((cx, cy))
                if roi_bboxes:
                    for roi_id, roi_box in enumerate(roi_bboxes):
                        overlap = self.get_overlap((x1, y1, x2, y2), roi_box)
                        track.update_overlap(roi_id, overlap)
                updated_tracks[matched_id] = track
                used_ids.add(matched_id)
            else:
                # новий трек
                new_track = Track(self.next_id, (cx, cy))
                updated_tracks[self.next_id] = new_track
                self.next_id += 1

        # треки без оновлення отримують "0" перекриття (маска зникла)
        for track_id, track in self.tracks.items():
            if track_id not in used_ids:
                if roi_bboxes:
                    for roi_id in range(len(roi_bboxes)):
                        track.update_overlap(roi_id, overlap=0.0)
                updated_tracks[track_id] = track

        self.tracks = updated_tracks
        return self.tracks

    def get_roi_statuses(self, num_rois):
        """
        агрегує статуси всіх треків для кожного ROI
        якщо хоча б один трек має "occupied" — ROI зайнятий
        """
        roi_statuses = {i: "free" for i in range(num_rois)}
        for track in self.tracks.values():
            for roi_id, status in track.status_per_roi.items():
                if status == "occupied":
                    roi_statuses[roi_id] = "occupied"
        return roi_statuses

    @staticmethod
    def get_overlap(bbox, roi_bbox):
        """
        обчислює відносну площу перекриття між bbox об'єкта і ROI
        """
        xA = max(bbox[0], roi_bbox[0])
        yA = max(bbox[1], roi_bbox[1])
        xB = min(bbox[2], roi_bbox[2])
        yB = min(bbox[3], roi_bbox[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        roi_area = (roi_bbox[2] - roi_bbox[0]) * (roi_bbox[3] - roi_bbox[1]) + 1e-6
        return inter_area / roi_area
