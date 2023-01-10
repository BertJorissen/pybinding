import copy
import numpy as np
from collections.abc import Iterable
from numpy.typing import ArrayLike
from typing import Optional

__all__ = ['FuzzySet']


class FuzzySet:
    def __init__(self, iterable: Optional[Iterable] = None, rtol: float = 1.e-3, atol: float = 1.e-5):
        self.data = []
        self.rtol = rtol
        self.atol = atol

        if iterable:
            for item in iterable:
                self.add(item)

    def __getitem__(self, index: int) -> ArrayLike:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, item) -> bool:
        return any(np.allclose(item, x, rtol=self.rtol, atol=self.atol) for x in self.data)

    def __iadd__(self, other) -> 'FuzzySet':
        for item in other:
            self.add(item)
        return self

    def __add__(self, other) -> 'FuzzySet':
        if isinstance(other, FuzzySet):
            ret = copy.copy(self)
            ret += other
            return ret
        else:
            return copy.copy(self)

    def __radd__(self, other) -> 'FuzzySet':
        return self + other

    def add(self, item) -> None:
        if item not in self:
            self.data.append(item)
