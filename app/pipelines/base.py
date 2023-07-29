from typing import Protocol

import pandas as pd


class PerYearPipeline(Protocol):
    def __call__(self, year: int) -> pd.DataFrame:
        ...
