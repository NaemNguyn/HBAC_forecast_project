import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from features import build_features

def train_model(grid: pd.DataFrame,
                tier_a: list,
                feat_cols: list) -> HistGradientBoostingRegressor:
    """Train ML model on Tier A SKUs."""
    X_rows, y_rows = [], []

    for i, sku in enumerate(tier_a):
        series = grid[sku]
        for d in pd.date_range('2024-03-01', '2025-09-04', freq='2D'):
            X_rows.append(build_features(d, series))
            y_rows.append(float(series.get(d, 0.0)))
        if (i + 1) % 20 == 0:
            print(f'  Features: {i+1}/{len(tier_a)} SKUs')

    X = pd.DataFrame(X_rows, columns=feat_cols).fillna(0)
    y = np.log1p(y_rows)

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04,
        max_leaf_nodes=47, min_samples_leaf=15,
        l2_regularization=0.5, random_state=42
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def ml_forecast(sku: str,
                grid: pd.DataFrame,
                model,
                feat_cols: list,
                val_dates: pd.DatetimeIndex,
                eval_dates: pd.DatetimeIndex) -> tuple:
    """Recursive ML forecast for one SKU."""
    series = grid[sku].copy()

    val_preds = []
    for d in val_dates:
        row  = build_features(d, series[series.index < d])
        Xrow = pd.DataFrame([row], columns=feat_cols).fillna(0)
        pred = max(0.0, float(np.expm1(model.predict(Xrow)[0])))
        val_preds.append(pred)
        series[d] = pred

    eval_preds = []
    for d in eval_dates:
        row  = build_features(d, series[series.index < d])
        Xrow = pd.DataFrame([row], columns=feat_cols).fillna(0)
        pred = max(0.0, float(np.expm1(model.predict(Xrow)[0])))
        eval_preds.append(pred)
        series[d] = pred

    return np.array(val_preds), np.array(eval_preds)


def dow_forecast(series: pd.Series,
                 dates: pd.DatetimeIndex,
                 n_weeks: int = 12) -> np.ndarray:
    """Exponentially weighted DOW average forecast."""
    s = series.copy()
    if series.iloc[-56:].sum() == 0:
        return np.zeros(len(dates))

    preds = []
    for d in dates:
        if d.dayofweek >= 5:
            preds.append(0.0)
            s[d] = 0.0
            continue
        past = s[s.index.dayofweek == d.dayofweek].iloc[-n_weeks:]
        if len(past) > 0 and past.sum() > 0:
            w    = np.exp(np.linspace(-2, 0, len(past)))
            pred = float(np.average(past.values, weights=w))
        else:
            pred = float(s.iloc[-28:].mean())
        pred = max(0.0, pred)
        preds.append(pred)
        s[d] = pred
    return np.array(preds)