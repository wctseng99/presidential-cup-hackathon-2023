import enum
from collections.abc import Callable, Iterable
from pathlib import Path

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


def get_deflation_data(data_dir: Path) -> pd.DataFrame:
    csv_path: Path = Path(data_dir, "DeflationCoefficient.csv")
    df = pd.read_csv(
        csv_path,
        index_col="year",
        usecols=["year", "DeflationCoef"],
    ).rename({"DeflationCoef": "deflation"}, axis=1)
    return df


def get_population_data(data_dir: Path) -> pd.DataFrame:
    csv_path: Path = Path(data_dir, "population.csv")
    df = pd.read_csv(
        csv_path,
        index_col="year",
        usecols=["year", "median"],
    ).rename({"median": "population"}, axis=1)
    return df


def get_vehicle_stock_data(data_dir: Path, vehicle_type: VehicleType) -> pd.DataFrame:
    vehicle: str = vehicle_type.value.lower()
    csv_path: Path = Path(data_dir, "stock", f"stock_{vehicle}.csv")
    df = pd.read_csv(
        csv_path,
        index_col="year",
        usecols=["year", "total"],
    ).rename({"total": "vehicle_stock"}, axis=1)
    return df


def get_vehicle_stock_adjustment_data(
    vehicle_type: VehicleType,
    extrapolate_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    match vehicle_type:
        case VehicleType.CAR:
            years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
            stock_adjustments = [
                0.2104,
                0.2164,
                0.2145,
                0.2164,
                0.2238,
                0.2298,
                0.2353,
                0.2369,
                0.2389,
            ]
        case VehicleType.SCOOTER:
            years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
            stock_adjustments = [
                0.4447,
                0.4533,
                0.4648,
                0.4721,
                0.4816,
                0.4873,
                0.4982,
                0.5013,
                0.5091,
            ]
        case _:
            raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    if extrapolate_years is not None:
        fn: Callable = lambda v, a, b: a + b * v
        (popt, _) = scipy.optimize.curve_fit(fn, years, stock_adjustments)

        for year in extrapolate_years:
            if year in years:
                continue

            years.append(year)
            stock_adjustments.append(fn(year, *popt))

    df = pd.DataFrame.from_dict(
        {"year": years, "vehicle_stock_adjustment": stock_adjustments}
    ).set_index("year")

    return df


def get_vehicle_ownership_data(
    data_dir: Path, vehicle_type: VehicleType
) -> pd.DataFrame:
    vehicle: str = vehicle_type.value.lower()
    csv_paths: Iterable[Path] = Path(data_dir, "GompertzData").glob("????.csv")

    df: pd.DataFrame
    dfs: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        year: int = int(csv_path.stem)

        df = pd.read_csv(csv_path).assign(year=year)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if vehicle_type in [VehicleType.CAR, VehicleType.SCOOTER]:
        columns: list[str] = ["year", "income", vehicle]
    else:
        raise ValueError(f"Invalid vehicle type: {vehicle_type}")

    df = df[columns]
    df = df.rename({vehicle: f"vehicle_ownership"}, axis=1).dropna()

    # income adjustment

    df_deflation: pd.DataFrame = get_deflation_data(data_dir=data_dir)
    df["income_adjusted"] = df.income / (
        df_deflation.deflation.loc[df.year].values / 100
    )

    # vehicle stock adjustments

    df_population: pd.DataFrame = get_population_data(data_dir=data_dir)
    df_vehicle_stock: pd.DataFrame = get_vehicle_stock_data(
        data_dir=data_dir, vehicle_type=vehicle_type
    )
    df_vehicle_stock_adjustment: pd.DataFrame = get_vehicle_stock_adjustment_data(
        vehicle_type=vehicle_type,
        extrapolate_years=df.year.unique(),
    )

    df_meta: pd.DataFrame = df_population.merge(df_vehicle_stock, on="year").merge(
        df_vehicle_stock_adjustment, on="year"
    )
    vehicle_stock_adjustment: pd.Series = df_meta.vehicle_stock / (
        df_meta.population * df_meta.vehicle_stock_adjustment
    )
    df["vehicle_ownership_adjusted"] = (
        df.vehicle_ownership * vehicle_stock_adjustment.loc[df.year].values
    )

    return df
