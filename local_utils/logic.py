import numpy as np

class SlotState:
    def __init__(self):
        self.status = "free"              # поточний статус: 'free', 'occupied', ...
        self.last_center = None           # центр останньої маски авто
        self.decay_counter = 0            # якщо використовуєш затухання

    def is_center_stable(self, new_center, threshold=30):
        """
        Перевіряє, чи новий центр близький до попереднього.
        """
        if self.last_center is None or new_center is None:
            return False
        dist = np.linalg.norm(np.array(self.last_center) - np.array(new_center))
        return dist < threshold