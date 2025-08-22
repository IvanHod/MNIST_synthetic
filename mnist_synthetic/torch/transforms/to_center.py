from typing import Union

import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torchvision.transforms import functional as F, InterpolationMode


ImgType = Union[NDArray[np.float32], torch.Tensor]


class ToCenterBase(torch.nn.Module):
    def __init__(self, offset: int | None = 4, resample: int | InterpolationMode | None = None) -> None:
        super().__init__()

        self._offset: int | None = offset
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

    def get_coords(self, img: ImgType, width: int, height: int) -> tuple[int, int, int, int] | None:
        row_nonzero = self.where((img > 0).any(axis=1))
        col_nonzero = self.where((img > 0).any(axis=0))

        if row_nonzero.shape[0] == 0 or col_nonzero.shape[0] == 0:
            return None

        top, bottom = row_nonzero[[0, -1]]
        bottom = min(bottom + 1, height)
        left, right = col_nonzero[[0, -1]]
        right = min(right + 1, width)

        return top, bottom, left, right

    def make_cut_area_to_square(self, img_cut: ImgType, coords: tuple[int, int, int, int]) -> ImgType:
        top, bottom, left, right = coords
        max_size = max(img_cut.shape[0], img_cut.shape[1])

        img_cut_squared = self.zeros((max_size, max_size), dtype=img_cut.dtype, device=img_cut.device)
        if img_cut.shape[0] == max_size:
            shift = (max_size - (right - left)) // 2
            img_cut_squared[:, shift: shift + (right - left)] = img_cut
        else:
            shift = (max_size - (bottom - top)) // 2
            img_cut_squared[shift:shift + bottom - top, :] = img_cut

        return img_cut_squared

    def insert_cut_to_source(
        self,
        img: ImgType,
        img_cut: ImgType,
        height: int,
        width: int,
    ) -> ImgType:
        result = self.zeros((height, width), dtype=img.dtype, device=img.device)

        _, h, w = self.normalize_shape(img_cut)
        offset_left = (width - w) // 2
        offset_top = (height - h) // 2

        if isinstance(img, torch.Tensor):
            result[offset_left: offset_left + w, offset_top: offset_top + h] = img_cut
        else:
            result[offset_top: offset_top + h, offset_left: offset_left + w] = img_cut

        return result

    def forward(self, img: ImgType, _label: int | None = None) -> ImgType:
        img_array, h, w = self.normalize_shape(img)

        coords = self.get_coords(img_array, h, w)
        if coords is None:
            return img

        top, bottom, left, right = coords

        img_cut = img_array[top: bottom, left: right]

        if self._offset is None:
            result = self.insert_cut_to_source(img, img_cut, h, w)
            return result[None,] if len(img.shape) == 3 else result

        img_cut_squared = self.make_cut_area_to_square(img_cut, coords)

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
