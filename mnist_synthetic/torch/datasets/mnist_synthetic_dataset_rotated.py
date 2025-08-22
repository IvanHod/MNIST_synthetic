from typing import Any, Optional, Callable, Literal
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from numpy._typing import NDArray

from torchvision.transforms import functional as F, InterpolationMode

from mnist_synthetic.generator import NumbersGenerator
from mnist_synthetic.torch.datasets import MNISTSynthetic


class MNISTSyntheticRotate(MNISTSynthetic):
    def __init__(
        self,
        size: int,
        max_angle: int,
        root: str | Path | None = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int | None = None,
        seed_angle: float | Literal['same'] | None = 'same',
        resample: int = cv2.INTER_NEAREST,
    ):
        self.max_angle = max_angle
        self.resample: int = resample
        super().__init__(size, root, transforms, transform, target_transform, seed)

        if seed_angle == 'same':
            seed_angle = seed

        self.random_angle = np.random.default_rng(seed=seed_angle)

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        img_np = self.data[idx]
        target: int = self.targets[idx]

        # Many transform methods required Pillow
        img = Image.fromarray(img_np, mode="L")
        if self.transform is not None:
            img = self.transform(img)

        angle = 0
        if self.max_angle > 0:
            angle = int(self.random_angle.integers(-self.max_angle, self.max_angle))
            img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            # img = rotation.rotate_image_np(img, angle, mode=self.resample)
            angle = -angle

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, angle


class MNISTSyntheticRotated(MNISTSyntheticRotate):
    """Like Validation Dataset with fixed rotations"""

    def _generate_image(self, generator: NumbersGenerator) -> tuple[NDArray[np.uint8], int, dict[str, Any] | None]:
        img, label = generator.generate()

        # img = ToCenterNumpy(offset=None, resample=self.resample)(img)

        angle = 0
        if self.max_angle > 0:
            # angle = generator.seed_rng.integers(-self.max_angle, self.max_angle)
            # img = rotation.rotate_image_np(img, angle, mode=self.resample)
            angle = -angle

        return img, int(label), {'angle': angle}

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        img, target = super(MNISTSyntheticRotate, self).__getitem__(idx)

        angle = self.extra[idx]['angle']
        if abs(angle) > 0:
            img = F.rotate(img, -angle, interpolation=InterpolationMode.BILINEAR)

        return img, target, self.extra[idx]['angle']
