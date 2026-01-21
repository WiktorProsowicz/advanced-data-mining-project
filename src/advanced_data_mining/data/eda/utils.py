"""Utilities for performing EDA or processed ds analysis."""

import pandas as pd

def is_outlier(series: pd.Series) -> pd.Series:
    """Determines what sentences are outliers using the IQR method."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)