from typing import Any, Optional, Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from numpy._typing import NDArray

from mnist_synthetic.generator import NumbersGenerator
from mnist_synthetic.torch.datasets import MNISTSynthetic
from mnist_synthetic.utils.rotation import Rotation


class MNISTSyntheticRotate(MNISTSynthetic):
    def __init__(
        self,
        size: int,
        root: str | Path | None = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int | None = None,
        rotate_before_np: Rotation | None = None,
        rotate_before_pill: Rotation | None = None,
        rotate_after: Rotation | None = None,
    ):
        assert sum(map(lambda v: v is not None, [rotate_before_np, rotate_before_pill, rotate_after])) == 1,\
            "This is Rotate module, rotate transform is required"

        self._rotate_before_np = rotate_before_np
        self._rotate_before_pill = rotate_before_pill
        self._rotate_after = rotate_after

        super().__init__(size, root, transforms, transform, target_transform, seed)

    def rotate_before_np(self, img: NDArray, angle: int | None = None) -> tuple[NDArray, int | None]:
        if self._rotate_before_np is None:
            return img, angle

        return self._rotate_before_np(img, angle=angle)

    def rotate_before_pill(self, img: Image.Image, angle: int  | None = None) -> tuple[Image.Image, int | None]:
        if self._rotate_before_pill is None:
            return img, angle

        return self._rotate_before_pill(img, angle=angle)

    def rotate_tensor(self, img: torch.Tensor, angle: int | None = None) -> tuple[torch.Tensor, int | None]:
        if self._rotate_after is None:
            return img, angle

        return self._rotate_after(img, angle=angle)

    @property
    def rotate_transform(self) -> Rotation:
        return self._rotate_before_np or self._rotate_before_pill or self._rotate_after

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        img_np = self.data[idx]
        target: int = self.targets[idx]

        img_np, angle = self.rotate_before_np(img_np)

        # Many transform methods required Pillow
        img = Image.fromarray(img_np, mode="L")
        img, angle = self.rotate_before_pill(img, angle=angle)
        if self.transform is not None:
            img = self.transform(img)

        img, angle = self.rotate_tensor(img, angle=angle)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, -angle


class MNISTSyntheticRotated(MNISTSyntheticRotate):
    """Like Validation Dataset with fixed rotations"""

    def _generate_image(self, generator: NumbersGenerator) -> tuple[NDArray[np.uint8], int, dict[str, Any] | None]:
        img, label = generator.generate()

        img, angle = self.rotate_before_np(img)
        if angle is None:
            angle = self.rotate_transform.generate_angle()

        assert angle is not None, 'Angle must be generated'
        return img, int(label), {'angle': angle}

    def __getitem__(self, idx: int) -> tuple[Any, Any, int]:
        img_np: NDArray[np.uint8] = self.data[idx]
        target: int = self.targets[idx]

        # Many transform methods required Pillow
        img = Image.fromarray(img_np, mode="L")
        img, angle = self.rotate_before_pill(img, angle=self.extra[idx]['angle'])

        if self.transform is not None:
            img = self.transform(img)

        img, angle = self.rotate_tensor(img, angle=angle)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, -angle
