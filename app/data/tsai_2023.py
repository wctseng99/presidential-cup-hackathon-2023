from pathlib import Path

import numpy as np
import pandas as pd

from app.data.core import (
    VehicleType,
    get_deflation_series,
    get_gdp_per_capita_series,
    get_population_series,
    get_vehicle_ownership_dataframe,
    get_vehicle_stock_adjustment_series,
    get_vehicle_stock_series,
)


def _get_gdp_data(
    data_dir: Path,
    index: pd.Index,
) -> pd.DataFrame:
    s_gdp_per_capita: pd.Series = get_gdp_per_capita_series(
        data_dir, extrapolate_index=index
    )
    s_deflation: pd.Series = get_deflation_series(
        data_dir=data_dir, extrapolate_index=index
    )

    df: pd.DataFrame = (
        pd.concat([s_gdp_per_capita, s_deflation], axis=1)
        .loc[index]
        .assign(
            adjusted_gdp_per_capita=lambda df: df.gdp_per_capita / (df.deflation / 100)
        )
    )

    return df


def get_tsai_sec_2_2_3_data(
    data_dir: Path,
    vehicle_type: VehicleType,
    income_bins: int = 100,
) -> pd.DataFrame:
    df_vehicle_ownership: pd.DataFrame = get_vehicle_ownership_dataframe(
        data_dir=data_dir, vehicle_type=vehicle_type
    )
    index: pd.Index = pd.Index(df_vehicle_ownership.year.sort_values().unique())

    # adjust income

    s_deflation: pd.Series = get_deflation_series(
        data_dir=data_dir, extrapolate_index=index
    )
    s_adjusted_income: pd.Series = df_vehicle_ownership.income / (
        s_deflation.loc[df_vehicle_ownership.year].values / 100
    )
    df_vehicle_ownership["adjusted_income"] = s_adjusted_income

    # adjust vehicle ownership

    s_population: pd.Series = get_population_series(
        data_dir=data_dir, extrapolate_index=index
    )
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        data_dir=data_dir, vehicle_type=vehicle_type, extrapolate_index=index
    )
    s_vehicle_stock_adjustment: pd.Series = get_vehicle_stock_adjustment_series(
        vehicle_type=vehicle_type, extrapolate_index=index
    )
    s: pd.Series = s_vehicle_stock / (s_population * s_vehicle_stock_adjustment)

    df_vehicle_ownership["adjusted_vehicle_ownership"] = (
        df_vehicle_ownership.vehicle_ownership * s.loc[df_vehicle_ownership.year].values
    )

    # bin by income

    s_income_bin: pd.Series = (
        s_adjusted_income.rank(pct=True)
        .mul(income_bins)
        .astype(int)
        .rename("income_bin")
    )
    df_vehicle_ownership_agg: pd.DataFrame = df_vehicle_ownership.groupby(
        s_income_bin
    ).agg(
        {"adjusted_income": np.mean, "adjusted_vehicle_ownership": np.mean}
    )  # type: ignore

    return df_vehicle_ownership_agg


def get_tsai_sec_2_3_data(
    data_dir: Path,
    vehicle_type: VehicleType,
) -> pd.DataFrame:
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        data_dir, vehicle_type=vehicle_type
    )
    index: pd.Index = s_vehicle_stock.index

    df_gdp: pd.DataFrame = _get_gdp_data(data_dir, index=index)
    df = pd.concat([s_vehicle_stock, df_gdp], axis=1).loc[index]

    return df


def get_tsai_sec_2_4_data(
    data_dir: Path,
    vehicle_type: VehicleType,
) -> pd.DataFrame:
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        data_dir, vehicle_type=vehicle_type
    )
    index: pd.Index = s_vehicle_stock.index

    s_population: pd.Series = get_population_series(data_dir, extrapolate_index=index)
    df_gdp: pd.DataFrame = _get_gdp_data(data_dir, index=index)

    df = pd.concat([s_vehicle_stock, s_population, df_gdp], axis=1).loc[index]
    df["log_gdp_per_capita"] = np.log(df.adjusted_gdp_per_capita)

    return df
