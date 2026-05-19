import pandas as pd
import numpy as np
from config import TRAIN_START, TRAIN_END

def build_grid(train: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : raw train.csv as DataFrame
    Output : dense grid (dates × SKUs) with net daily quantity
    """
    # Net quantity — returns handled naturally
    daily = (
        train.groupby(['Date', 'ItemCode'])['Quantity']
        .sum()
        .clip(lower=0)
        .reset_index()
        .rename(columns={'Quantity': 'Qty'})
    )
    daily = daily[daily['Qty'] > 0]

    all_dates = pd.date_range(TRAIN_START, TRAIN_END, freq='D')
    all_skus  = sorted(train['ItemCode'].unique())

    grid = (
        daily.pivot_table(index='Date', columns='ItemCode',
                          values='Qty', fill_value=0)
        .reindex(index=all_dates, columns=all_skus, fill_value=0)
    )
    return grid


def get_tiers(grid: pd.DataFrame) -> tuple:
    """
    Returns TIER_A, TIER_B, TIER_C lists based on activity
    """
    from config import (TIER_A_QTY_90D, TIER_A_QTY_365D,
                        TIER_B_QTY_90D, TIER_B_QTY_365D)

    qty_90d  = grid[grid.index >= '2025-06-07'].sum()
    qty_365d = grid[grid.index >= '2024-09-06'].sum()

    TIER_A = qty_90d[
        (qty_90d  >= TIER_A_QTY_90D) &
        (qty_365d >= TIER_A_QTY_365D)
    ].index.tolist()

    TIER_B = qty_90d[
        (qty_90d  >= TIER_B_QTY_90D) &
        (qty_365d >= TIER_B_QTY_365D) &
        (~qty_90d.index.isin(TIER_A))
    ].index.tolist()

    TIER_C = [s for s in grid.columns
              if s not in TIER_A and s not in TIER_B]

    return TIER_A, TIER_B, TIER_C