import numpy as np
import pandas as pd

def wrmsse(pred: pd.DataFrame,
           actual: pd.DataFrame,
           train: pd.DataFrame,
           weights: pd.Series) -> float:
    """
    Weighted Root Mean Squared Scaled Error.
    Lower is better.
    """
    mse        = ((pred.values - actual.values) ** 2).mean(axis=0)
    train_mean = train.iloc[-28:].mean(axis=0).values
    scale      = train_mean ** 2 + 1e-8
    rmsse      = np.sqrt(mse / scale)
    w          = weights.values / weights.sum()
    return float((w * rmsse).sum())