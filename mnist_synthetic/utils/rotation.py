import cv2
from numpy._typing import NDArray


def rotate_image_np(img: NDArray, angle: int, interpolation: int = cv2.INTER_NEAREST) -> NDArray:
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((h // 2, w // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=interpolation)
