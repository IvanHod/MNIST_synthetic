from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    width: int = 28
    height: int = 28
    channels: int = 1

    draw_color: tuple[int, int, int] = (255, 255, 255)
    draw_thickness: int = 2