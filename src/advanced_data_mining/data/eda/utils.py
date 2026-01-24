"""Utilities for performing EDA or processed ds analysis."""

from typing import Any

import pandas as pd
import seaborn as sns


def is_outlier(series: pd.Series) -> pd.Series:
    """Determines what sentences are outliers using the IQR method."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)


def get_gradient_palette(n: int) -> Any:
    """Returns a gradient colormap used for EDA plots."""

    return sns.color_palette("blend:#2a5957,#5dc2bd", n_colors=n)
