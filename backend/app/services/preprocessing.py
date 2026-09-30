from __future__ import annotations

import pandas as pd


def inspect_dataset(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("The dataset is empty.")
    missing_ratio = float(frame.isna().mean().mean())
    duplicate_ratio = float(frame.duplicated().mean())
    quality = round(max(0, min(100, 100 - missing_ratio * 45 - duplicate_ratio * 25)))
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "quality_score": quality,
        "missing_values": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated().sum()),
        "column_names": list(frame.columns),
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }
