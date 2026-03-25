from typing import Optional, Union, TYPE_CHECKING

import cv2
import numpy as np
from numpy._typing import NDArray

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode


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


def tv_rotate_tensor(img: "torch.Tensor", angle: int, mode: Optional["InterpolationMode"] = None) -> "torch.Tensor":
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    if mode is None:
        mode = InterpolationMode.BICUBIC

    return F.rotate(img, angle, interpolation=mode)


class Rotation:
    def __init__(
        self,
        max_angle: int,
        interpolation: Optional[Union[int, "Image.Resampling", "InterpolationMode"]] = None,
        seed: int | None = None,
    ) -> None:
        self.max_angle = max_angle
        self.interpolation = interpolation
        self.angle_generator = np.random.default_rng(seed=seed)

    def generate_angle(self) -> int:
        if self.max_angle == 0:
            return 0

        return int(self.angle_generator.integers(-self.max_angle, self.max_angle))

    def __call__(
            self,
            img: Union[NDArray, "Image", "torch.Tensor"],
            angle: int | None = None
    ) -> tuple[Union[NDArray, "Image", "torch.Tensor"], int | None]:
        if angle is None:
            angle = self.generate_angle()

        if isinstance(img, np.ndarray):
            if self.interpolation is None:
                self.interpolation = cv2.INTER_LINEAR
            assert isinstance(self.interpolation, int), ('For numpy Image interpolation must cv2 constants')
            img_rotated = rotate_image_np(img, angle=angle, mode=self.interpolation)
        else:
            from PIL import Image
            import torch
            from torchvision.transforms import InterpolationMode

            if isinstance(img, Image.Image):
                if self.interpolation is None:
                    self.interpolation = Image.Resampling.BILINEAR
                assert isinstance(self.interpolation, Image.Resampling), ('For pillow Image interpolation must '
                                                                          'be Image.Resampling')
                img_rotated = img.rotate(angle, resample=self.interpolation)
            elif isinstance(img, torch.Tensor):
                if self.interpolation is None:
                    self.interpolation = InterpolationMode.BILINEAR
                assert isinstance(self.interpolation, InterpolationMode), ('For tensor Image interpolation must '
                                                                           'be InterpolationMode')
                img_rotated = tv_rotate_tensor(img, angle=angle, mode=self.interpolation)
            else:
                raise TypeError(f"Unsupported type {type(img)}")

        return img_rotated, angle