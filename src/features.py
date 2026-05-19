import pandas as pd
import numpy as np
from config import LAG_DAYS, ROLL_WINDOWS, DOW_LAG_WKS

def build_features(d: pd.Timestamp, series: pd.Series) -> dict:
    """
    Build one feature row for date d using history before d.
    """
    hist = series[series.index < d]
    row  = {
        'dow'           : d.dayofweek,
        'month'         : d.month,
        'dom'           : d.day,
        'quarter'       : d.quarter,
        'week'          : int(d.isocalendar().week),
        'is_weekend'    : int(d.dayofweek >= 5),
        'is_tet'        : int((d.month==1 and d.day>=15) or
                              (d.month==2 and d.day<=15)),
        'is_month_end'  : int(d.day >= 25),
        'is_month_start': int(d.day <= 5),
    }
    # Point lags
    for lag in LAG_DAYS:
        row[f'lag_{lag}'] = float(
            series.get(d - pd.Timedelta(days=lag), 0.0))

    # Rolling stats
    for w in ROLL_WINDOWS:
        win = hist.iloc[-w:] if len(hist) >= w else hist
        row[f'rmean_{w}'] = float(win.mean())
        row[f'rstd_{w}']  = float(win.std()) if len(win) > 1 else 0.
        row[f'rmax_{w}']  = float(win.max())
        row[f'rpos_{w}']  = float((win > 0).mean())

    # Same weekday lags
    same_dow = hist[hist.index.dayofweek == d.dayofweek]
    for wk in DOW_LAG_WKS:
        row[f'dlag_{wk}w'] = float(same_dow.iloc[-wk]) \
                              if len(same_dow) >= wk else 0.

    # Trend
    t28 = hist.iloc[-28:]
    row['trend_28'] = float(
        np.polyfit(np.arange(len(t28)),
                   t28.values.astype(float), 1)[0]
    ) if len(t28) > 2 else 0.

    # Zero inflation
    row['zero_frac_28'] = float(
        (t28 == 0).mean()) if len(t28) > 0 else 1.

    # Same period last year
    ly     = d - pd.DateOffset(years=1)
    ly_win = hist[
        (hist.index >= ly - pd.Timedelta(days=14)) &
        (hist.index <= ly + pd.Timedelta(days=14))
    ]
    row['ly_mean'] = float(ly_win.mean()) if len(ly_win) > 0 else 0.

    return row