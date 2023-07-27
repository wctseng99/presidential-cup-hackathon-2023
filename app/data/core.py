import enum
import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize


class VehicleType(str, enum.Enum):
    CAR = "CAR"
    SCOOTER = "SCOOTER"
    OPERATING_CAR = "OPERATING_CAR"
    BUS = "BUS"
    TRUCK = "TRUCK"
    LIGHT_TRUCK = "LIGHT_TRUCK"
    HEAVY_TRUCK = "HEAVY_TRUCK"


class City(str, enum.Enum):
    NEW_TAIPEI = "NEW_TAIPEI"
    TAIPEI = "TAIPEI"
    TAOYUAN = "TAOYUAN"
    TAICHUNG = "TAICHUNG"
    TAINAN = "TAINAN"
    KAOHSIUNG = "KAOHSIUNG"
    YILAN = "YILAN"
    HSINCHU = "HSINCHU"
    MIAOLI = "MIAOLI"
    CHANGHUA = "CHANGHUA"
    NANTOU = "NANTOU"
    YUNLIN = "YUNLIN"
    CHIAYI = "CHIAYI"
    PINGTUNG = "PINGTUNG"
    TAITUNG = "TAITUNG"
    HUALIEN = "HUALIEN"
    PENGHU = "PENGHU"
    KEELUNG = "KEELUNG"
    HSINCHU_CITY = "HSINCHU_CITY"
    CHIAYI_CITY = "CHIAYI_CITY"
    JINMA = "JINMA"


def to_camel_case(s: str) -> str:
    return "".join(word.capitalize() for word in s.lower().split("_"))


def to_snake_case(s: str, upper: bool = True) -> str:
    s = "".join([("_" if c.isupper() else "") + c for c in s]).lstrip("_")
    return s.upper() if upper else s.lower()


DEFAULT_YEAR_INDEX: pd.Index = pd.RangeIndex(1990, 2051, name="year")


def extrapolate_series(
    s: pd.Series,
    index: pd.Index,
    fn: Callable | str = lambda v, a, b: a + b * v,
    use_original: bool = True,
) -> pd.Series:
    s_extrapolated: pd.Series

    if callable(fn):
        popt, _ = scipy.optimize.curve_fit(fn, s.index.values, s.values)
        values: np.ndarray = np.vectorize(fn)(index.values, *popt)

        s_extrapolated = pd.Series(values, index=index)

        if use_original:
            s_extrapolated = s.combine_first(s_extrapolated)

    elif isinstance(fn, str):
        s_extrapolated = (
            pd.Series(index=index, dtype=s.dtype)
            .combine_first(s)
            .interpolate(method=fn)
        )

    else:
        raise NotImplementedError

    return s_extrapolated


def get_column_data_fn(
    csv_name: str,
    index_column: str,
    value_column: str | None = None,
    value_column_rename: str | None = None,
    extrapolate_method: Callable | str = lambda v, a, b: a + b * v,
    available_value_columns: Iterable[str] | None = None,
) -> Callable[..., pd.Series]:
    def get_column_data(
        data_dir: Path,
        # in some cases, we want to load other columns other than the default `value_column`
        value_column: str | None = value_column,
        extrapolate_index: pd.Index | None = None,
        extrapolate_method: Callable | str = extrapolate_method,
        extrapolate_use_original: bool = True,
    ) -> pd.Series:
        if value_column is None:
            raise ValueError(f"No value_column specified for csv_name={csv_name}. ")

        if (
            available_value_columns is not None
            and value_column not in available_value_columns
        ):
            raise ValueError(
                f"Invalid value_column={value_column} for csv_name={csv_name}. "
                f"Available value_columns={available_value_columns}"
            )

        csv_path: Path = Path(data_dir, csv_name)
        df: pd.DataFrame = pd.read_csv(
            csv_path, index_col=index_column, usecols=[index_column, value_column]
        )
        s: pd.Series = df[value_column]
        if value_column_rename is not None:
            s = s.rename(value_column_rename)

        if extrapolate_index is not None:
            s = extrapolate_series(
                s,
                index=extrapolate_index,
                fn=extrapolate_method,
                use_original=extrapolate_use_original,
            )

        return s

    return get_column_data


get_deflation_series = get_column_data_fn(
    csv_name="DeflationCoefficient.csv",
    index_column="year",
    value_column="DeflationCoef",
    value_column_rename="deflation",
)

get_income_series = get_column_data_fn(
    csv_name="income.csv",
    index_column="year",
    value_column="total",
    value_column_rename="income",
    available_value_columns=itertools.chain(["total"], map(to_camel_case, City)),
)

get_gdp_per_capita_series = get_column_data_fn(
    csv_name="GDPperCapita.csv",
    index_column="year",
    value_column="GDPperCapita",
    value_column_rename="gdp_per_capita",
)

get_gini_series = get_column_data_fn(
    csv_name="giniIndex.csv",
    index_column="year",
    value_column="gini",
    extrapolate_method="pad",
)

get_population_series = get_column_data_fn(
    csv_name="population.csv",
    index_column="year",
    value_column="median",
    value_column_rename="population",
)

get_city_population_series = get_column_data_fn(
    csv_name="population_city.csv",
    index_column="year",
    value_column_rename="population",
)


def get_city_area_series(data_dir: Path) -> pd.Series:
    _get_city_area_series: Callable[..., pd.Series] = get_column_data_fn(
        csv_name="cityArea.csv",
        index_column="city",
        value_column="area",
    )
    s: pd.Series = _get_city_area_series(data_dir=data_dir)
    s.index = s.index.map(to_snake_case)

    return s


def get_vehicle_survival_rate_series(
    data_dir: Path,
    vehicle_type: VehicleType,
) -> pd.Series:
    if vehicle_type not in [
        VehicleType.CAR,
        VehicleType.SCOOTER,
        VehicleType.OPERATING_CAR,
    ]:
        raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    vehicle: str = vehicle_type.value.lower()
    _get_vehicle_survival_rate_series: Callable[..., pd.Series] = get_column_data_fn(
        csv_name="survival_rate_original.csv",
        index_column="age",
        value_column=vehicle,
        value_column_rename="survival_rate",
    )
    return _get_vehicle_survival_rate_series(data_dir=data_dir)


def get_vehicle_stock_series(
    data_dir: Path,
    vehicle_type: VehicleType,
    city: City | None = None,
    extrapolate_index: pd.Index | None = None,
) -> pd.Series:
    if vehicle_type not in [
        VehicleType.CAR,
        VehicleType.SCOOTER,
        VehicleType.OPERATING_CAR,
        VehicleType.BUS,
        VehicleType.TRUCK,
        VehicleType.LIGHT_TRUCK,
        VehicleType.HEAVY_TRUCK,
    ]:
        raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    vehicle: str = vehicle_type.value.lower()
    city_str: str = "total" if city is None else to_camel_case(city)

    _get_vehicle_stock_series: Callable[..., pd.Series] = get_column_data_fn(
        csv_name=f"stock/stock_{vehicle}.csv",
        index_column="year",
        value_column=city_str,
        value_column_rename="vehicle_stock",
    )
    return _get_vehicle_stock_series(
        data_dir=data_dir, extrapolate_index=extrapolate_index
    )


def get_vehicle_stock_adjustment_series(
    vehicle_type: VehicleType, extrapolate_index: pd.Index | None = None
) -> pd.Series:
    match vehicle_type:
        case VehicleType.CAR:
            data = {
                2010: 0.2104,
                2011: 0.2164,
                2012: 0.2145,
                2013: 0.2164,
                2014: 0.2238,
                2015: 0.2298,
                2016: 0.2353,
                2017: 0.2369,
                2018: 0.2389,
            }

        case VehicleType.SCOOTER:
            data = {
                2010: 0.4447,
                2011: 0.4533,
                2012: 0.4648,
                2013: 0.4721,
                2014: 0.4816,
                2015: 0.4873,
                2016: 0.4982,
                2017: 0.5013,
                2018: 0.5091,
            }
        case _:
            raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    s = pd.Series(data, name="vehicle_stock_adjustment")
    s.index.name = "year"

    if extrapolate_index is not None:
        s = extrapolate_series(s, index=extrapolate_index)
    return s


def get_income_dataframe(
    data_dir: Path,
    index: pd.Index | None = DEFAULT_YEAR_INDEX,
) -> pd.DataFrame:
    s_income: pd.Series = get_income_series(data_dir=data_dir, extrapolate_index=index)
    s_gini: pd.Series = get_gini_series(data_dir=data_dir, extrapolate_index=index)
    s_deflation: pd.Series = get_deflation_series(
        data_dir=data_dir, extrapolate_index=index
    )

    df: pd.DataFrame = pd.concat([s_income, s_gini, s_deflation], axis=1).dropna()
    if index is not None:
        df = df.loc[index]

    df["adjusted_income"] = df.income / (df.deflation / 100)

    return df


def get_gdp_dataframe(
    data_dir: Path,
    index: pd.Index | None = DEFAULT_YEAR_INDEX,
) -> pd.DataFrame:
    s_gdp_per_capita: pd.Series = get_gdp_per_capita_series(
        data_dir, extrapolate_index=index
    )
    s_deflation: pd.Series = get_deflation_series(
        data_dir=data_dir, extrapolate_index=index
    )

    df: pd.DataFrame = (
        pd.concat([s_gdp_per_capita, s_deflation], axis=1).loc[index].sort_index()
    )
    df["adjusted_gdp_per_capita"] = df.gdp_per_capita / (df.deflation / 100)

    return df


def get_city_population_dataframe(
    data_dir: Path,
    cities: Iterable[City] = City,
    extrapolate_index: pd.Index | None = DEFAULT_YEAR_INDEX,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for city in cities:
        s_population: pd.Series = get_city_population_series(
            data_dir=data_dir,
            value_column=to_camel_case(city),
            extrapolate_index=extrapolate_index,
        )
        dfs.append(s_population.reset_index().assign(city=city.value))

    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    df = df.set_index(["city", "year"]).sort_index()

    return df


def get_vehicle_ownership_dataframe(
    data_dir: Path,
    vehicle_type: VehicleType,
) -> pd.DataFrame:
    if vehicle_type not in [VehicleType.CAR, VehicleType.SCOOTER]:
        raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    vehicle: str = vehicle_type.value.lower()
    csv_paths: Iterable[Path] = Path(data_dir, "GompertzData").glob("????.csv")

    dfs: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        year: int = int(csv_path.stem)

        df: pd.DataFrame = pd.read_csv(csv_path, index_col=["id"])
        if vehicle not in df.columns:
            continue

        df = df[["income", vehicle]].rename({vehicle: "vehicle_ownership"}, axis=1)
        df = df.assign(year=year)

        dfs.append(df)

    df = pd.concat(dfs)

    return df
