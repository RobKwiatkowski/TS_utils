import math
import numpy as np
import pytest

import utils


def test_macd():
    data = np.array([1, 1, 1, 1])
    result = utils.macd(data)
    expected = np.array([0, 0, 0, 0])

    assert all([a == b for a, b in zip(result, expected)])


def test_rsi():
    data = np.arange(0, 4, 1)
    result = utils.rsi(data, 3)
    expected = np.array([np.NaN, np.NaN, 100.0, 100.0])

    assert all([math.isnan(a) for a in result[:1]])  # check for NaN
    assert all([pytest.approx(a) == b for a, b in zip(result[2:], expected[2:])])  # check for values
