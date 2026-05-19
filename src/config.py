import pandas as pd

# Dates
TRAIN_START    = pd.Timestamp('2020-11-17')
TRAIN_END      = pd.Timestamp('2025-09-05')
VAL_DATES      = pd.date_range('2025-09-06', '2025-10-03', freq='D')
EVAL_DATES     = pd.date_range('2025-10-04', '2025-10-31', freq='D')
WEIGHT_START   = pd.Timestamp('2025-08-09')

# Feature constants
LAG_DAYS       = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
ROLL_WINDOWS   = [7, 14, 28, 56, 90]
DOW_LAG_WKS    = [1, 2, 4, 8, 12]

# Tier thresholds
TIER_A_QTY_90D  = 150
TIER_A_QTY_365D = 400
TIER_B_QTY_90D  = 30
TIER_B_QTY_365D = 80

# Paths
TRAIN_PATH      = '../data/raw/train.csv'
GRID_PATH       = '../data/processed/daily_sales.parquet'
SAMPLE_PATH     = '../data/raw/sample_submission.csv'
MODEL_PATH      = '../models/hgbr_model.pkl'
SUBMISSION_PATH = '../data/submission/submission_final.csv'