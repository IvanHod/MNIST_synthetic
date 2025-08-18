from abc import ABC, abstractmethod

import numpy as np

Generator = np.random._generator.Generator

class BaseLocation(ABC, object):
    @abstractmethod
    def generate(self, seed_rng: Generator, _prev_val: int | None = None) -> int:
        raise NotImplementedError()


class IntLocation(BaseLocation):
    def __init__(self, value: int):
        self._value = value

    def generate(self, seed_rng: Generator, _prev_val: int | None = None) -> int:
        return self._value


class RelatedLocation(BaseLocation):
    def __init__(self, value: float, max_size: int):
        self._value: float = value
        self._max_size: int = max_size

    def generate(self, seed_rng: Generator, _prev_val: int | None = None) -> int:
        return int(self._value * self._max_size)


class RangeLocation(BaseLocation):
    def __init__(self, start: int | float, end: int | float):
        self.start: int | float = start
        self.end: int | float = end

    def generate(self, seed_rng: Generator, _prev_val: int | None = None) -> int:
        assert isinstance(self.start, int) and isinstance(self.end, int), 'start and end must be integers'
        return int(seed_rng.integers(self.start, self.end))


class RangeRelatedLocation(RangeLocation):
    def __init__(self, start: float, end: float, max_size: int):
        super().__init__(start, end)
        self._max_size = max_size

    def generate(self, seed_rng: Generator, _prev_val: int | None = None) -> int:
        assert isinstance(self.start, float) and isinstance(self.end, float), 'start and end must be floats'
        start, end = int(self.start * self._max_size), int(self.end * self._max_size)
        if start == end:
            return start

        return int(seed_rng.integers(start, end))


class RangeOnPrevLocation(BaseLocation):
    def __init__(self, offset_neg: int | float, offset_pos: int | float, max_size: int):
        self.offset_neg: int | float = offset_neg
        self.offset_pos: int | float = offset_pos

        self.max_size: int = max_size

    def generate(self, seed_rng: Generator, prev_val: int | None = None) -> int:
        assert isinstance(self.offset_neg, int) and isinstance(self.offset_pos, int), 'start and end must be integers'
        assert prev_val is not None, 'RangeOnPrevLocation requires a previous value'

        res = seed_rng.integers(max(0, prev_val - self.offset_neg), min(self.max_size, prev_val + self.offset_pos))
        return int(res)


class RangeOnPrevRelatedLocation(RangeOnPrevLocation):
    def generate(self, seed_rng: Generator, prev_val: int | None = None) -> int:
        assert isinstance(self.offset_neg, float) and 0 <= abs(self.offset_neg) <= 1, 'offset_neg must be between 0 and 1'
        assert isinstance(self.offset_pos, float) and 0 <= abs(self.offset_pos) <= 1, 'offset_pos must be between 0 and 1'
        assert prev_val is not None, 'RangeOnPrevLocation requires a previous value'

        offset_neg = int(self.offset_neg * self.max_size)
        offset_pos = int(self.offset_pos * self.max_size)
        res = seed_rng.integers(max(0, prev_val - offset_neg), min(self.max_size, prev_val + offset_pos))
        return int(res)