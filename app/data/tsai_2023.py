from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.core import (
    City,
    VehicleType,
    get_city_area_series,
    get_city_population_dataframe,
    get_deflation_series,
    get_gdp_dataframe,
    get_population_series,
    get_vehicle_ownership_dataframe,
    get_vehicle_stock_adjustment_series,
    get_vehicle_stock_series,
)


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

    df_gdp: pd.DataFrame = get_gdp_dataframe(data_dir, index=index)
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
    df_gdp: pd.DataFrame = get_gdp_dataframe(data_dir, index=index)

    df = pd.concat([s_vehicle_stock, s_population, df_gdp], axis=1).loc[index]
    df["log_gdp_per_capita"] = np.log(df.adjusted_gdp_per_capita)

    return df


def get_tsai_sec_2_5_data(
    data_dir: Path,
    vehicle_type: VehicleType,
    exclude_cities: set[City] = set([City.JINMA]),
) -> pd.DataFrame:
    df_vehicle_stock: pd.DataFrame
    df_vehicle_stocks: list[pd.DataFrame] = []
    for city in City:
        if city in exclude_cities:
            continue

        s_vehicle_stock: pd.Series = get_vehicle_stock_series(
            data_dir, vehicle_type=vehicle_type, city=city
        )
        df_vehicle_stocks.append(s_vehicle_stock.reset_index().assign(city=city.value))

    df_vehicle_stock = pd.concat(df_vehicle_stocks, axis=0)
    cities: Iterable[City] = map(City, df_vehicle_stock.city.unique())
    years: Iterable[int] = df_vehicle_stock.year.sort_values().unique()

    s_city_area: pd.Series = get_city_area_series(data_dir)
    df_population: pd.DataFrame = get_city_population_dataframe(
        data_dir, cities=cities, extrapolate_index=pd.Index(years, name="year")
    )

    df = df_vehicle_stock.merge(s_city_area, on="city", how="left").merge(
        df_population, on=["year", "city"], how="left"
    )
    df["population_density"] = df.population / df.area
    df["vehicle_stock_density"] = df.vehicle_stock / df.population

    return df
