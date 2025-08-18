import numpy as np

from mnist_synthetic.config import GeneratorConfig
from mnist_synthetic.generator import NumbersGenerator


def test_generate_1_as_line():
    """Check generation only horizontal line"""
    config = GeneratorConfig(width=10, height=10, draw_thickness=1)
    generator = NumbersGenerator(seed=3, config=config)
    img, label = generator.generate(include=['1'])

    np.testing.assert_array_equal(
        img[:8, 2:7],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8) * 255
    )


def test_generate_1_check_seed():
    """Check generation only horizontal line"""
    config = GeneratorConfig(width=10, height=10, draw_thickness=1)
    generator = NumbersGenerator(seed=3, config=config)
    img, label = generator.generate(include=['1'])

    np.testing.assert_array_equal(
        img[:8, 2:7],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8) * 255
    )

    img, label = generator.generate(include=['1'])
    np.testing.assert_array_equal(
        img[:8, :5],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8) * 255
    )


def test_generate_1_with_peak():
    """Check generation only horizontal line"""
    config = GeneratorConfig(width=10, height=10, draw_thickness=1)
    generator = NumbersGenerator(seed=1, config=config)
    img, label = generator.generate(include=['1'])

    np.testing.assert_array_equal(
        img[:8, 1:6],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8) * 255
    )


def test_generate_1_with_root():
    config = GeneratorConfig(width=10, height=10, draw_thickness=1)
    generator = NumbersGenerator(seed=4, config=config)

    img, label = generator.generate(include=['1'])

    np.testing.assert_array_equal(
        img[:9, 1:7],
        np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8) * 255
    )
