import numpy as np
import pandas as pd

from typing import Union


def macd(data: Union[np.ndarray, pd.DataFrame], fast: int = 12, slow: int = 28):
    """

    Args:
        data: time series data
        fast: period of the fast EWA
        slow: period of the slow EWA

    Returns: array of a calculated MACD

    """
    try:
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
    except TypeError:
        print("Wrong data input type")

    exp1 = df.ewm(span=fast, adjust=False).mean()
    exp2 = df.ewm(span=slow, adjust=False).mean()
    return np.array(exp1 - exp2)


def rsi(data: Union[np.ndarray, pd.DataFrame], window: int):
    try:
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
    except TypeError:
        print("Wrong data input type")

    change = df.diff()
    change_up = change[change>0].fillna(0)
    change_down = change[change<0].fillna(0)
    avg_up = change_up.rolling(window).mean()
    avg_down = change_down.rolling(window).mean().abs()
    rsi_result = 100 * avg_up / (avg_up + avg_down)
    return rsi_result.to_numpy().flatten()

