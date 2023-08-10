from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.core import (
    City,
    Vehicle,
    get_city_area_series,
    get_city_population_dataframe,
    get_column_data_fn,
    get_deflation_series,
    get_gdp_dataframe,
    get_population_series,
    get_vehicle_ownership_dataframe,
    get_vehicle_stock_adjustment_series,
    get_vehicle_stock_series,
    get_vehicle_survival_rate_series,
)


def get_tsai_vehicle_survival_rate_series(
    data_dir: Path,
    result_dir: Path,
    vehicle: Vehicle,
    max_age: int = 25,
) -> pd.Series:
    vehicle_str: str = vehicle.value.lower()
    _get_column_data_fn = get_column_data_fn(
        csv_name=f"tsai-2023-sec-2-2-1-{vehicle_str}.csv",
        index_column="age",
        value_column="survival_rate",
    )
    s_result: pd.Series = _get_column_data_fn(data_dir=result_dir)
    s_result = s_result.reindex(pd.RangeIndex(max_age + 1, name="age"), fill_value=0)

    s = get_vehicle_survival_rate_series(data_dir=data_dir, vehicle=vehicle)

    return pd.Series.combine_first(s_result, s)


def get_tsai_vehicle_stock_series(
    result_dir: Path,
    vehicle: Vehicle,
    percentage: float | None = None,
) -> pd.Series:
    vehicle_str: str = vehicle.value.lower()

    df: pd.DataFrame = pd.read_csv(
        Path(result_dir, f"tsai-2023-sec-3-1-{vehicle_str}.csv"),
        index_col=["year", "percentage"],
        usecols=["year", "percentage", "vehicle_stock"],
    )

    s: pd.Series
    if percentage:
        df = df.reset_index()
        df = df.loc[df["percentage"] == percentage]
        s = df.set_index("year")["vehicle_stock"]
    else:
        s = df["vehicle_stock"]

    return s


def get_tsai_sec_2_2_3_data(
    data_dir: Path,
    vehicle: Vehicle,
    income_bins: int = 100,
) -> pd.DataFrame:
    df_vehicle_ownership: pd.DataFrame = get_vehicle_ownership_dataframe(
        data_dir=data_dir, vehicle=vehicle
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
        data_dir=data_dir, vehicle=vehicle, extrapolate_index=index
    )
    s_vehicle_stock_adjustment: pd.Series = get_vehicle_stock_adjustment_series(
        vehicle=vehicle, extrapolate_index=index
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
    vehicle: Vehicle,
    extrapolate_index: pd.Index | None = None,
) -> pd.DataFrame:
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        data_dir=data_dir, vehicle=vehicle, extrapolate_index=extrapolate_index
    )
    if extrapolate_index is None:
        extrapolate_index = s_vehicle_stock.index

    df_gdp: pd.DataFrame = get_gdp_dataframe(
        data_dir=data_dir, extrapolate_index=extrapolate_index
    )
    df = pd.concat([s_vehicle_stock, df_gdp], axis=1).loc[extrapolate_index]

    return df


def get_tsai_sec_2_4_data(
    data_dir: Path,
    vehicle: Vehicle,
    extrapolate_index: pd.Index | None = None,
) -> pd.DataFrame:
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        data_dir=data_dir, vehicle=vehicle, extrapolate_index=extrapolate_index
    )
    if extrapolate_index is None:
        extrapolate_index = s_vehicle_stock.index

    s_population: pd.Series = get_population_series(
        data_dir=data_dir, extrapolate_index=extrapolate_index
    )
    df_gdp: pd.DataFrame = get_gdp_dataframe(
        data_dir=data_dir, extrapolate_index=extrapolate_index
    )

    df = pd.concat([s_vehicle_stock, s_population, df_gdp], axis=1).loc[
        extrapolate_index
    ]
    df["log_gdp_per_capita"] = np.log(df.adjusted_gdp_per_capita)

    return df


def get_tsai_sec_2_5_data(
    data_dir: Path,
    vehicle: Vehicle,
    cities: Iterable[City],
    extrapolate_index: pd.Index | None = None,
) -> pd.DataFrame:
    df_vehicle_stock: pd.DataFrame
    df_vehicle_stocks: list[pd.DataFrame] = []
    for city in cities:
        s_vehicle_stock: pd.Series = get_vehicle_stock_series(
            data_dir=data_dir,
            vehicle=vehicle,
            city=city,
            extrapolate_index=extrapolate_index,
        )
        df_vehicle_stocks.append(s_vehicle_stock.reset_index().assign(city=city.value))

    df_vehicle_stock = pd.concat(df_vehicle_stocks, axis=0)
    years: Iterable[int] = df_vehicle_stock.year.sort_values().unique()

    s_city_area: pd.Series = get_city_area_series(data_dir=data_dir)
    df_population: pd.DataFrame = get_city_population_dataframe(
        data_dir=data_dir, cities=cities, extrapolate_index=pd.Index(years, name="year")
    )

    df = df_vehicle_stock.merge(s_city_area, on="city", how="left").merge(
        df_population, on=["year", "city"], how="left"
    )
    df["population_density"] = df.population / df.area
    df["vehicle_stock_density"] = df.vehicle_stock / df.population

    return df
