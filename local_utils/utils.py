import numpy as np


def get_mask_center(mask):
    """
    Обчислює центр маски як середнє значення всіх ненульових пікселів.
    """
    if mask is None or mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    return np.array([xs.mean(), ys.mean()])