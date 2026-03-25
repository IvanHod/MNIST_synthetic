import math
from abc import ABC, abstractmethod
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from mnist_synthetic.config import GeneratorConfig
from mnist_synthetic.utils.drawing import Drawing
from mnist_synthetic.location import (
    IntLocation,
    RangeLocation,
    RangeOnPrevLocation,
    RelatedLocation,
    RangeRelatedLocation,
    RangeOnPrevRelatedLocation,
    RangeRelatedValueLocation
)


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, include: list[str] | None = None) -> tuple[NDArray[np.uint8], str]:
        raise NotImplementedError()



class NumbersGenerator(BaseGenerator):
    ALLOWS_SYMBOLS: ClassVar[tuple[str, ...]] = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def __init__(self, seed: int | None = None, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()

        self.seed_rng = np.random.default_rng(seed=seed)
        self.draw: Drawing = Drawing(self.seed_rng, self.config.draw_color, self.config.draw_thickness)

    def _init_image(self):
        img = np.zeros((self.config.height, self.config.height), dtype=np.uint8)
        if self.config.channels > 1:
            img = img[None, ]
            img = np.repeat(img, self.config.channels, axis=0)

        return img


    def generate(self, include: list[str] | None = None) -> tuple[NDArray[np.uint8], str]:
        if include is None:
            include = self.ALLOWS_SYMBOLS

        label: str = self.seed_rng.choice(include, size=1)[0]
        img = getattr(self, f'generate_{label}')()

        return img, label

    def generate_0(self) -> NDArray[np.uint8]:
        img = self._init_image()
        _ = self.draw.draw_ellipse(
            img,
            center_x_location=RelatedLocation(0.5, max_size=self.config.width),
            center_y_range=RelatedLocation(0.5, max_size=self.config.height),
            width_location=RangeRelatedLocation(0.15, 0.36, max_size=self.config.width),
            height_location=RangeRelatedLocation(0.25, 0.45, max_size=self.config.height),
            angle_location=RangeLocation(-1, 1),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        return img

    def generate_1(self):
        width: int = self.config.width

        img = self._init_image()
        left, top, right, bottom = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.3, 0.6, max_size=width),
            top_location=RangeRelatedLocation(0.15, 0.36, max_size=self.config.height),
            right_location=RangeOnPrevRelatedLocation(0.1, 0., max_size=width),
            bottom_location=RangeRelatedLocation(0.65, 0.8, max_size=self.config.height),
        )

        if self.seed_rng.random() < 0.2:
            self.draw.draw_line(
                img,
                left_location=IntLocation(left),
                top_location=IntLocation(top),
                right_location=RangeOnPrevRelatedLocation(0.3, -0.15, max_size=width),
                bottom_location=RangeOnPrevRelatedLocation(-0.15, 0.3, max_size=self.config.height),
            )

        if self.seed_rng.random() < 0.2:
            self.draw.draw_line(
                img,
                left_location=RangeLocation(right - int(0.22 * width), right - int(0.1 * width)),
                top_location=IntLocation(bottom),
                right_location=RangeLocation(right + int(0.1 * width), right + int(0.22 * width)),
                bottom_location=IntLocation(bottom),
            )

        return img

    def generate_2(self):
        img = self._init_image()
        center_x, center_y, width, height, angle = self.draw.draw_ellipse(
            img,
            center_x_location=RelatedLocation(0.5, max_size=self.config.width),
            center_y_range=RelatedLocation(0.28, max_size=self.config.width),
            width_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.width),
            height_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.width),
            angle_location=RangeLocation(-30, 30),
            start_angle_location=IntLocation(180),
            end_angle_location=IntLocation(410),
        )

        left = int(np.where(img.max(axis=0) > 0)[0][0])
        bottom = int(center_y + np.where(img[center_y:, center_x:].max(axis=1) == 0)[0][0] - 1)
        right = int(np.where(img[bottom, :] > 0)[0][-1])

        _, _, right_, bottom_ = self.draw.draw_line(
            img,
            left_location=IntLocation(right),
            top_location=IntLocation(bottom),
            right_location=RangeLocation(left, right),
            bottom_location=RangeLocation(center_y + int(0.3 * self.config.height),
                                          center_y + int(0.5 * self.config.width)),
        )
        self.draw.draw_line(
            img,
            left_location=IntLocation(right_),
            top_location=IntLocation(bottom_),
            right_location=RangeLocation(right_ + int(0.3 * self.config.width),
                                         max(right_ + int(0.5 * self.config.width), right)),
            bottom_location=RangeOnPrevLocation(1, 0, max_size=self.config.height),
        )

        return img

    def generate_3(self):
        img = self._init_image()
        center_x, center_y, width, height, angle = self.draw.draw_ellipse(
            img,
            center_x_location=RelatedLocation(0.5, max_size=self.config.width),
            center_y_range=RelatedLocation(0.3, max_size=self.config.width),
            width_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.width),
            height_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.width),
            angle_location=RangeLocation(-10, 40),
            start_angle_location=IntLocation(180),
            end_angle_location=IntLocation(420),
        )

        left = np.where(img.max(axis=0) > 0)[0][0]

        bottom = center_y + np.where(img[center_y:, center_x:].max(axis=1) == 0)[0][0] - 1
        right = np.where(img[bottom, :] > 0)[0][-1]

        img2 = np.zeros((center_x, self.config.width), dtype='uint8')
        self.draw.draw_ellipse(
            img2,
            center_x_location=RelatedLocation(0.3, max_size=self.config.width),
            center_y_range=RelatedLocation(0.3, max_size=self.config.width),
            width_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.width),
            height_location=RangeRelatedLocation(0.15, 0.22, max_size=self.config.height),
            angle_location=RangeLocation(-10, 20),
            start_angle_location=IntLocation(270),
            end_angle_location=IntLocation(480),
        )

        img2 = img2[np.where(img2 > 0)[0][0] + 2:]  # режем верхние черные пиксели
        left2 = np.where(img2[0, :] > 0)[0][0]
        img2 = np.roll(img2, right - left2, axis=1)
        img[bottom:bottom + img2.shape[0], :] = img2

        x1 = left
        y1 = np.where(img[:, x1] > 0)[0].max()
        x2 = np.where(img[bottom + 5:, :].max(axis=0) > 0)[0].min()
        y2 = np.where(img[:, x2 + 1] > 0)[0].max()

        if x2 - x1 < 3:
            return img

        center_y, center_x = int(0.5 * self.config.height), int(0.5 * self.config.width)
        angle_deg = math.degrees(math.atan2(x2 - x1, y2 - y1))  # важно: (dx, -dy)
        M = cv2.getRotationMatrix2D((center_y, center_x), -angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return rotated

    def generate_4(self) -> np.ndarray:
        if self.config.digit_4_version <= 0:
            return self.generate_4_top_open()
        elif self.config.digit_4_version >= 1:
            return self.generate_4_top_closed()

        # Вероятностный выбор
        if np.random.rand() < self.config.digit_4_version:
            img = self.generate_4_top_closed()
        else:
            img = self.generate_4_top_open()
        return img

    def generate_4_top_open(self) -> np.ndarray:
        # 4 - here top is open
        img = self._init_image()

        left, top, right, middle = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.1, 0.5, max_size=self.config.width),
            top_location=RangeRelatedLocation(0.1, 0.3, max_size=self.config.height),
            right_location=RangeOnPrevRelatedLocation(0.15, 0., max_size=self.config.width),
            bottom_location=RangeOnPrevRelatedLocation(-0.2, 0.5, max_size=self.config.height),
        )
        right, _, _, _ = self.draw.draw_line(
            img,
            left_location=RangeLocation(left + 4, left + 8),
            top_location=RangeLocation(1, 8),
            right_location=RangeOnPrevLocation(1, 0, max_size=self.config.width),
            bottom_location=RangeLocation(middle + 7, middle + 12),
        )

        _ = self.draw.draw_line(
            img,
            left_location=IntLocation(left),
            top_location=IntLocation(middle),
            right_location=IntLocation(right),
            bottom_location=IntLocation(middle),
        )

        return img

    def generate_4_top_closed(self) -> np.ndarray:
        img = self._init_image()

        # straight right line
        r_left, r_top, r_right, r_bottom = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.6, 0.9, max_size=self.config.width),
            top_location=RangeRelatedLocation(0.1, 0.3, max_size=self.config.height),
            right_location=RangeOnPrevRelatedLocation(0.1, 0.1, max_size=self.config.width),
            bottom_location=RangeOnPrevRelatedLocation(-0.5, 0.8, max_size=self.config.height),
        )


        # diagonal line to the left
        c_left, c_bottom, _, _ = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.2, 0.9, max_size=r_left),
            top_location=RangeRelatedValueLocation(0.3, 0.8, max_size=r_bottom - r_top, value=r_top),
            right_location=IntLocation(r_left),
            bottom_location=IntLocation(r_top),
        )

        # horizontal line to the right crossed vertical line
        self.draw.draw_line(
            img,
            left_location=IntLocation(c_left),
            top_location=IntLocation(c_bottom),
            right_location=RangeRelatedValueLocation(0.05, 0.2, max_size=self.config.width, value=r_left),
            bottom_location=RangeOnPrevRelatedLocation(0.1, 0.1, max_size=self.config.height),
        )

        return img

    def generate_5(self):
        img = self._init_image()
        left, top, right, bottom = self.draw.draw_line(
            img,
            left_location=RangeLocation(9, 10),
            top_location=RangeLocation(4, 5),
            right_location=RangeLocation(15, 20),
            bottom_location=RangeOnPrevLocation(4, 0, max_size=self.config.height),
        )
        _, _, right, bottom = self.draw.draw_line(
            img,
            left_location=IntLocation(left),
            top_location=IntLocation(top),
            right_location=RangeOnPrevLocation(4, 0, max_size=self.config.width),
            bottom_location=RangeLocation(max(top, bottom) + 5, max(top, bottom) + 10)
        )

        img2 = self._init_image()
        _ = self.draw.draw_ellipse(
            img2,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(15),
            width_location=RangeLocation(4, 7),
            height_location=RangeLocation(4, 7),
            angle_location=RangeLocation(-1, 1),
            start_angle_location=IntLocation(230),
            end_angle_location=IntLocation(480)
        )

        c_left = np.where(img2[:15].max(axis=0) > 0)[0][0]
        c_top = np.where(img2[:, c_left] > 0)[0][0]

        img2 = np.roll(img2, (bottom - c_top, right - c_left), axis=(0, 1))
        img[img2 > 0] = img2[img2 > 0]

        return img

    def generate_6(self):
        img = self._init_image()

        center_x, center_y, width, height, angle = self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(18),
            width_location=RangeLocation(4, 7),
            height_location=RangeLocation(4, 7),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360)
        )
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(18),
            width_location=IntLocation(width),
            height_location=RangeLocation(height + 7, height + 12),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(180),
            end_angle_location=RangeLocation(250, 300)
        )

        return img

    def generate_7(self):
        img = self._init_image()

        left, top, right, t_bottom = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.1, 0.2, self.config.width),
            top_location=RangeRelatedLocation(0.2, 0.3, self.config.height),
            right_location=RangeRelatedLocation(0.2, 0.8, self.config.width),
            # upper or a little bit lower
            bottom_location=RangeRelatedLocation(0.0, 0.32, self.config.height)
        )

        _, _, right_, bottom_ = self.draw.draw_line(
            img,
            left_location=IntLocation(right),
            top_location=IntLocation(t_bottom),
            # From the left start of digit, and a little bit at the right
            right_location=RangeOnPrevRelatedLocation(left / self.config.width, 0.05, self.config.width),
            # Down almost to the bottom border
            bottom_location=RangeOnPrevRelatedLocation(-0.4, 0.6, self.config.height)
        )

        if self.config.digit_7_h_line_proba >= np.random.rand():
            # if 1. -> then digit_7_h_line_proba will be bigger or equal then np.random.rand()
            middle = (t_bottom + bottom_) // 2
            center = (right + right_) // 2
            self.draw.draw_line(
                img,
                left_location=RangeRelatedValueLocation(-0.2, -0.02, self.config.width, value=center),
                top_location=IntLocation(middle),
                right_location=RangeRelatedValueLocation(0.02, 0.2, self.config.width, value=center),
                bottom_location=IntLocation(middle)
            )

        return img

    def generate_8(self):
        img = self._init_image()

        center_x, center_y, width, height, angle = self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(8),
            width_location=RangeLocation(3, 7),
            height_location=RangeLocation(3, 6),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360)
        )
        h2 = np.random.randint(height, height + 3)
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(8 + height + h2),
            width_location=RangeLocation(width, width + 3),
            height_location=IntLocation(h2),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360)
        )

        return img
    
    def generate_9(self):
        img = self._init_image()
    
        center_x, center_y, width, height, angle = self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14),
            center_y_range=IntLocation(8),
            width_location=RangeLocation(4, 8),
            height_location=RangeLocation(4, 8),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360)
        )
        
        w = np.random.randint(width, width + 5)
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(14 - (w - width)),
            center_y_range=IntLocation(8),
            width_location=IntLocation(w),
            height_location=IntLocation(15),
            angle_location=RangeLocation(-10, 10),
            start_angle_location=IntLocation(0),
            end_angle_location=RangeLocation(90, 120)
        )
    
        return img
