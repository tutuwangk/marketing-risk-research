"""Year-based train/valid/test split."""
import numpy as np

from .config import TEST_YEARS, TRAIN_YEARS, VALID_YEARS, YEAR


def split_masks(df):
    """Boolean masks over df rows for train, valid, test and refit (train + valid), by 报告年度."""
    year = df[YEAR].to_numpy()
    masks = {
        "train": np.isin(year, list(TRAIN_YEARS)),
        "valid": np.isin(year, list(VALID_YEARS)),
        "test": np.isin(year, list(TEST_YEARS)),
    }
    masks["refit"] = masks["train"] | masks["valid"]
    return masks
