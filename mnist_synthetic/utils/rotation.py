from typing import Optional

import cv2
from numpy._typing import NDArray


def rotate_image_np(img: NDArray, angle: int, mode: int = cv2.INTER_NEAREST) -> NDArray:
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((h // 2, w // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=mode)


def tv_rotate_image_np(img: NDArray, angle: int, mode: Optional["InterpolationMode.INTER_NEAREST"] = None) -> NDArray:
    from torchvision import transforms as tv
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    if mode is None:
        mode = InterpolationMode.BICUBIC

    img_tensor = tv.ToTensor()(img)

    return F.rotate(img_tensor, angle, interpolation=mode)


def tv_rotate_tensor(img: "torch.Tensor", angle: int, mode: Optional["InterpolationMode.INTER_NEAREST"] = None) -> "torch.Tensor":
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    if mode is None:
        mode = InterpolationMode.BICUBIC

    return F.rotate(img, angle, interpolation=mode)
