import pandas as pd
import numpy as np

def build_submission(forecast_cache: dict,
                     sku_order: list,
                     sample: pd.DataFrame) -> pd.DataFrame:
    """
    Build submission DataFrame matching sample_submission.csv exactly.
    """
    F_COLS = [f'F{i}' for i in range(1, 29)]
    ZEROS  = np.zeros(28)
    rows   = []

    for sku in sku_order:
        vp, _ = forecast_cache.get(sku, (ZEROS, ZEROS))
        rows.append([f'{sku}_validation'] + np.maximum(vp, 0).tolist())

    for sku in sku_order:
        _, ep = forecast_cache.get(sku, (ZEROS, ZEROS))
        rows.append([f'{sku}_evaluation'] + np.maximum(ep, 0).tolist())

    submission = pd.DataFrame(rows, columns=['id'] + F_COLS)

    # Validation checks
    assert list(submission['id']) == list(sample['id']), 'ID mismatch!'
    assert len(submission) == 31944,                      'Wrong rows!'
    assert (submission[F_COLS].values >= 0).all(),        'Negatives!'

    return submission
