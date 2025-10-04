"""Minimal numpy stub for tests.
This is not a full implementation and only supports features required by agent.MiniTfidf."""

from __future__ import annotations

import math
import random as _random
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]


class ndarray:
    def __init__(self, data):
        if isinstance(data, ndarray):
            self._data = data.tolist()
        else:
            self._data = data

    @property
    def ndim(self) -> int:
        if not self._data:
            return 1
        first = self._data[0]
        return 2 if isinstance(first, list) else 1

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.ndim == 1:
            return (len(self._data),)
        return (len(self._data), len(self._data[0]) if self._data else 0)

    def tolist(self):
        if self.ndim == 1:
            return list(self._data)
        return [list(row) for row in self._data]

    def _apply(self, func):
        if self.ndim == 1:
            return ndarray([func(v) for v in self._data])
        return ndarray([[func(v) for v in row] for row in self._data])

    def _binary(self, other, func):
        if isinstance(other, ndarray):
            other_data = other.tolist()
        else:
            other_data = other
        if isinstance(other_data, list):
            if self.ndim == 1:
                return ndarray([func(a, b) for a, b in zip(self._data, other_data)])
            return ndarray([
                [func(a, b) for a, b in zip(row, other_row)]
                for row, other_row in zip(self._data, other_data)
            ])
        return self._apply(lambda v: func(v, other_data))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self._data[i][j]
        return self._data[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            i, j = idx
            self._data[i][j] = value
        else:
            self._data[idx] = value

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        if isinstance(other, ndarray) and self.ndim == 2 and other.ndim == 1:
            return ndarray([
                [value * other._data[idx] for idx, value in enumerate(row)]
                for row in self._data
            ])
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary(other, lambda a, b: b / a)

    def __neg__(self):
        return self._apply(lambda v: -v)

    def __imatmul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(other)
        if self.ndim == 2 and other.ndim == 1:
            result = []
            for row in self._data:
                result.append(sum(a * b for a, b in zip(row, other._data)))
            return ndarray(result)
        raise NotImplementedError("matmul not supported for shapes")

    def __imul__(self, other):
        updated = self.__mul__(other)
        self._data = updated.tolist()
        return self


float32 = float


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        data: Union[List[float], List[List[float]]] = [0.0] * shape
    else:
        rows, cols = shape
        data = [[0.0] * cols for _ in range(rows)]
    return ndarray(data)


def log(value):
    if isinstance(value, ndarray):
        return value._apply(math.log)
    return math.log(value)


def array(seq):  # pragma: no cover - helper, not used directly
    return ndarray(list(seq))


def argsort(seq):
    if isinstance(seq, ndarray):
        data = seq.tolist()
    else:
        data = list(seq)
    return [index for index, _ in sorted(enumerate(data), key=lambda item: item[1])]


class _LinalgModule:
    @staticmethod
    def norm(arr, axis=None):
        if isinstance(arr, ndarray):
            data = arr.tolist()
        else:
            data = arr
        if axis is None:
            if data and isinstance(data[0], list):
                total = sum(value * value for row in data for value in row)
            else:
                total = sum(value * value for value in data)
            return math.sqrt(total)
        if axis == 1:
            return ndarray([math.sqrt(sum(value * value for value in row)) for row in data])
        raise NotImplementedError("norm only supports axis=None or axis=1")


linalg = _LinalgModule()


class _RandomModule:
    @staticmethod
    def seed(value):  # pragma: no cover - deterministic helper
        _random.seed(value)


random = _RandomModule()
