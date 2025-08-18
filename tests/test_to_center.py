import cv2
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import ToTensor

from mnist_synthetic.torch.transforms.to_center import ToCenterNumpy, ToCenter


def test_numpy_zeros():
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255
    img_out = ToCenterNumpy(offset=1)(img)
    assert (img_out == 0).all()


def test_numpy_ones():
    img = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.uint8) * 255
    img_out = ToCenterNumpy(offset=1)(img)
    np.testing.assert_array_equal(img_out, np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255)


def test_numpy_to_center():
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255

    img_out = ToCenterNumpy(offset=1)(img)
    np.testing.assert_array_equal(img_out, np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255)


def test_numpy_to_center_interp_knn():
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255

    img_out = ToCenterNumpy(offset=1, resample=cv2.INTER_NEAREST)(img)
    np.testing.assert_array_equal(img_out, np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255)


def test_torch_to_center():
    img = ToTensor()(np.zeros((5, 5), dtype=np.float32))
    img[0, 1, 1] = 1

    img_out = ToCenter(offset=1, resample=InterpolationMode.NEAREST)(img)
    img_true = ToTensor()(np.zeros((5, 5), dtype=np.float32))
    img_true[0, 1: 4, 1: 4] = 1

    torch.testing.assert_close(img_out, img_true)


def test_torch_to_center_interp():
    img = ToTensor()(np.zeros((5, 5), dtype=np.float32))
    img[0, 1:3, 1] = 1

    img_out = ToCenter(offset=1, resample=InterpolationMode.NEAREST)(img)
    img_true = ToTensor()(np.zeros((5, 5), dtype=np.float32))
    img_true[0, 1: 4, 1: 3] = 1.

    torch.testing.assert_close(img_out, img_true)
