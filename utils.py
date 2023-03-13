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
