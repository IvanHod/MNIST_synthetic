from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    width: int = 28
    height: int = 28
    channels: int = 1

    draw_color: tuple[int, int, int] = (255, 255, 255)
    draw_thickness: int = 2

    digit_4_version: float = 0.0  # 0.0 - is open 4, 1.0 - closed (triangle) 4; 0.5 - both
    digit_7_h_line_proba: float = 1.0  # By default, it is always drawing