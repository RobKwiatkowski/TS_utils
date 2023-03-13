import pytest
import numpy as np
import utils


def test_macd():
    data = np.array([1, 1, 1, 1])
    result = utils.macd(data)
    expected = np.array([0, 0, 0, 0])
    assert all([a == b for a, b in zip(result, expected)])
