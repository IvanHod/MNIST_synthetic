import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torchvision.transforms import functional as F, InterpolationMode


class ToCenterBase(torch.nn.Module):
    def __init__(self, offset: int = 4, resample: int | InterpolationMode | None = None) -> None:
        super().__init__()

        self._offset = offset
        self.resample: int | InterpolationMode = self.set_resample(resample)

    def set_resample(self, resample: int | InterpolationMode | None) -> int | InterpolationMode:
        raise NotImplementedError()

    @classmethod
    def normalize_shape(cls, img):
        raise NotImplementedError()

    @classmethod
    def where(cls, exp):
        raise NotImplementedError()

    @classmethod
    def zeros(cls, shape, dtype, device):
        raise NotImplementedError()

    @classmethod
    def resize(cls, img, shape, resample):
        raise NotImplementedError()

    def forward(self, img: NDArray | torch.Tensor, _label: int | None = None):
        img_array, h, w = self.normalize_shape(img)

        row_nonzero = self.where((img_array > 0).any(axis=1))
        col_nonzero = self.where((img_array > 0).any(axis=0))

        if row_nonzero.shape[0] == 0 or col_nonzero.shape[0] == 0:
            return img

        top, bottom = row_nonzero[[0, -1]]
        bottom = min(bottom + 1, h)
        left, right = col_nonzero[[0, -1]]
        right = min(right + 1, w)

        img_cut = img_array[top: bottom, left: right]
        max_size = max(img_cut.shape[0], img_cut.shape[1])

        img_cut_squared = self.zeros((max_size, max_size), dtype=img.dtype, device=img.device)
        if img_cut.shape[0] == max_size:
            shift = (max_size - (right - left)) // 2
            img_cut_squared[:, shift: shift + (right - left)] = img_cut
        else:
            shift = (max_size - (bottom - top)) // 2
            img_cut_squared[shift:shift + bottom - top, :] = img_cut

        img_size = (img.shape[-1] - self._offset * 2)
        img_cut_squared = self.resize(img_cut_squared, [img_size, img_size], self.resample)

        result = self.zeros((h, w), dtype=img_cut_squared.dtype, device=img.device)
        result[self._offset: self._offset + img_size, self._offset: self._offset + img_size] = img_cut_squared
        return result[None,] if len(img.shape) == 3 else result


class ToCenterNumpy(ToCenterBase):
    def set_resample(self, resample: int | None) -> int:
        if resample is None:
            return cv2.INTER_NEAREST

        return resample

    @classmethod
    def normalize_shape(cls, img):
        return img, img.shape[0], img.shape[1]

    @classmethod
    def where(cls, exp):
        return np.where(exp)[0]

    @classmethod
    def zeros(cls, shape: tuple[int, int], dtype, device):
        return np.zeros(shape, dtype=dtype)

    @classmethod
    def resize(cls, img, shape: tuple[int, int], resample: int):
        return cv2.resize(img, shape, interpolation=resample)


class ToCenter(ToCenterBase):
    def set_resample(self, resample: InterpolationMode | None) -> InterpolationMode:
        if resample is None:
            return InterpolationMode.NEAREST

        return resample

    @classmethod
    def normalize_shape(cls, img):
        return img[0], img.shape[-1], img.shape[-2]

    @classmethod
    def where(cls, exp):
        return torch.where(exp)[0]

    @classmethod
    def zeros(cls, shape: tuple[int, int], dtype, device):
        return torch.zeros(shape, dtype=dtype).to(device)

    @classmethod
    def resize(cls, img, shape, resample):
        return F.resize(img[None, ], shape, interpolation=resample)[0]
