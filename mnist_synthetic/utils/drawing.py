import cv2
import numpy as np
from numpy._typing import NDArray

from mnist_synthetic.location import BaseLocation


class Drawing:
    def __init__(self, seed_rng: np.random._generator.Generator, color: tuple[int, int, int], thickness: int):
        self.seed_rng = seed_rng

        self.color = color
        self.thickness = thickness

    def draw_line(
        self,
        img: NDArray[np.uint8],
        top_location: BaseLocation,
        left_location: BaseLocation,
        bottom_location: BaseLocation,
        right_location: BaseLocation,
    ) -> tuple[int, int, int, int]:
        left = left_location.generate(self.seed_rng)
        top = top_location.generate(self.seed_rng)
        right = right_location.generate(self.seed_rng, left)
        bottom = bottom_location.generate(self.seed_rng, top)

        cv2.line(img, (left, top), (right, bottom), color=self.color, thickness=self.thickness)

        return left, top, right, bottom

    def draw_ellipse(
        self,
        img: NDArray[np.uint8],
        center_x_location: BaseLocation,
        center_y_range: BaseLocation,
        width_location: BaseLocation,
        height_location: BaseLocation,
        angle_location: BaseLocation,
        start_angle_location: BaseLocation,
        end_angle_location: BaseLocation,
    ) -> tuple[int, int, int, int, int]:
        center_x = center_x_location.generate(self.seed_rng)
        center_y = center_y_range.generate(self.seed_rng)

        width = width_location.generate(self.seed_rng)
        height = height_location.generate(self.seed_rng)

        angle = angle_location.generate(self.seed_rng)

        start_angle = start_angle_location.generate(self.seed_rng)
        end_angle = end_angle_location.generate(self.seed_rng)

        cv2.ellipse(img, (center_x, center_y), (width, height), angle, start_angle, end_angle,
                    color=self.color, thickness=self.thickness)

        return center_x, center_y, width, height, angle