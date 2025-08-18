# MNIST Synthetic

The library for digits generations (like MNIST). The default size of images is 28x28.

![generated_rotate_numbers.png](https://raw.githubusercontent.com/IvanHod/MNIST_synthetic/refs/heads/master/assets/generated_rotate_numbers.png)

## Number Generation

You can use default settings, or configurate generated number use `GeneratorConfig`.
```python
from mnist_synthetic.generator import NumbersGenerator
from mnist_synthetic.config import GeneratorConfig

config: GeneratorConfig = GeneratorConfig()
generator = NumbersGenerator(seed=None, )
```

To generate number it's enough to run method
```python
img, label = generator.generate_0()
```

To generate random number call just `generate`:
```python
img, label = generator.generate()
```

## Datasets

To run it:
```python
from matplotlib import pyplot as plt
from mnist_synthetic.torch.datasets import MNISTSynthetic

dataset = MNISTSynthetic(10, seed=42)

fig, axes = plt.subplots(1, 10, figsize=(10, 4))
for i, ax in enumerate(axes):
    ax.imshow(dataset[i][0], cmap='gray')
    ax.axis('off')
    ax.set_label(str(dataset[i][1]))
```

