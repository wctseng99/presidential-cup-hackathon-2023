import dataclasses
import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rich.progress
import scipy.integrate
import sympy as sp
from absl import logging

from app.data.core import (
    City,
    Fuel,
    Vehicle,
    get_gini_series,
    get_income_dataframe,
    get_population_series,
    get_vehicle_age_composition_series,
    get_vehicle_market_share_series,
)
from app.data.tsai_2023 import (
    get_tsai_sec_2_2_3_data,
    get_tsai_sec_2_3_data,
    get_tsai_sec_2_4_data,
    get_tsai_sec_2_5_data,
    get_tsai_vehicle_stock_series,
    get_tsai_vehicle_survival_rate_series,
)
from app.modules.base import BootstrapModule
from app.modules.core import LinearModule
from app.modules.tsai_2023 import (
    BusStockDensityModule,
    CarOwnershipModuleV2,
    IncomeDistributionModule,
    OperatingCarStockModule,
    ScooterOwnershipModule,
    TruckStockModule,
    VehicleAgeCompositionModule,
)
from app.pipelines.base import PerYearPipeline


@dataclasses.dataclass(kw_only=True)
class CarStockPipeline(PerYearPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.CAR

    data_dir: Path
    income_bins_total: int = 100
    income_bins_removed: int = 1
    bootstrap_fit_runs: int = 300
    bootstrap_predict_runs: int = 300
    integrate_sigma: float = 64
    quantiles: Iterable[float] = (0.025, 0.5, 0.975)

    income_module: LinearModule = dataclasses.field(default_factory=LinearModule)
    income_distribution_module: IncomeDistributionModule = dataclasses.field(
        default_factory=IncomeDistributionModule
    )
    vehicle_ownership_module: CarOwnershipModuleV2 = dataclasses.field(
        default_factory=CarOwnershipModuleV2
    )

    @functools.cached_property
    def df_vehicle_ownership(self) -> pd.DataFrame:
        return get_tsai_sec_2_2_3_data(
            self.data_dir, vehicle=self.VEHICLE, income_bins=self.income_bins_total
        )

    @functools.cached_property
    def bootstrap_income_module(self) -> BootstrapModule:
        df_income: pd.DataFrame = get_income_dataframe(self.data_dir)

        bootstrap_income_module: BootstrapModule = BootstrapModule(
            module=self.income_module, runs=self.bootstrap_fit_runs
        )
        bootstrap_income_module.fit(
            X=np.r_["1,2,0", df_income.index.values],
            y=df_income["adjusted_income"].values,
        )
        return bootstrap_income_module

    @functools.cached_property
    def bootstrap_vehicle_ownership_module(self) -> BootstrapModule:
        df_vehicle_ownership_to_fit: pd.DataFrame = self.df_vehicle_ownership.head(
            -self.income_bins_removed
        ).tail(-self.income_bins_removed)

        bootrap_vehicle_ownership_module: BootstrapModule = BootstrapModule(
            module=self.vehicle_ownership_module, runs=self.bootstrap_fit_runs
        )
        bootrap_vehicle_ownership_module.fit(
            income=df_vehicle_ownership_to_fit["adjusted_income"].values,
            ownership=df_vehicle_ownership_to_fit["adjusted_vehicle_ownership"].values,
        )
        return bootrap_vehicle_ownership_module

    def __call__(self, year: int) -> pd.DataFrame:
        index: pd.Index = pd.Index([year], name="year")

        df_income: pd.DataFrame = get_income_dataframe(self.data_dir)
        s_gini: pd.Series = get_gini_series(self.data_dir, extrapolate_index=index)
        s_population: pd.Series = get_population_series(
            self.data_dir, extrapolate_index=index
        )

        vehicle_ownership_vals: list[float] = []
        for _ in range(self.bootstrap_predict_runs):
            mean_income: float
            if year in df_income.index:
                mean_income = df_income.loc[year, "adjusted_income"]
            else:
                mean_income = self.bootstrap_income_module(
                    output=self.income_module.y, x_0=year, run_one=True
                )

            income_pdf = self.income_distribution_module(
                output=self.income_distribution_module.income_pdf,
                mean_income=mean_income,
                gini=s_gini.loc[year],
            )
            ownership_var = self.bootstrap_vehicle_ownership_module(
                output=self.vehicle_ownership_module.ownership,
                income=self.income_distribution_module.income_var,
                run_one=True,
            )

            fn: Callable = sp.lambdify(
                self.income_distribution_module.income_var,
                ownership_var * income_pdf,
            )
            vehicle_ownership_val, _ = scipy.integrate.quad(
                fn, 0, self.integrate_sigma * mean_income
            )
            vehicle_ownership_vals.append(vehicle_ownership_val)

        df: pd.DataFrame = (
            pd.Series(vehicle_ownership_vals, name="adjusted_vehicle_ownership")
            .quantile(self.quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .astype(np.float32)
        )
        df["vehicle_stock"] = (
            df["adjusted_vehicle_ownership"].mul(s_population[year]).astype(int)
        )

        return df


@dataclasses.dataclass(kw_only=True)
class ScooterStockPipeline(CarStockPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.SCOOTER

    vehicle_ownership_module: ScooterOwnershipModule = dataclasses.field(  # type: ignore
        default_factory=ScooterOwnershipModule
    )


@dataclasses.dataclass(kw_only=True)
class OperatingCarStockPipeline(PerYearPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.OPERATING_CAR

    data_dir: Path
    bootstrap_fit_runs: int = 1000
    bootstrap_predict_runs: int = 1000
    quantiles: Iterable[float] = (0.025, 0.5, 0.975)

    vehicle_stock_module: OperatingCarStockModule = dataclasses.field(
        default_factory=OperatingCarStockModule
    )

    @functools.cached_property
    def bootstrap_vehicle_stock_module(self) -> BootstrapModule:
        df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_3_data(
            self.data_dir, vehicle=self.VEHICLE
        )

        bootstrap_vehicle_stock_module: BootstrapModule = BootstrapModule(
            module=self.vehicle_stock_module, runs=self.bootstrap_fit_runs
        )
        bootstrap_vehicle_stock_module.fit(
            gdp_per_capita=df_vehicle_stock["adjusted_gdp_per_capita"].values,
            vehicle_stock=df_vehicle_stock["vehicle_stock"].values,
        )
        return bootstrap_vehicle_stock_module

    def __call__(self, year: int) -> pd.DataFrame:
        index: pd.Index = pd.Index([year], name="year")

        df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_3_data(
            self.data_dir, vehicle=self.VEHICLE, extrapolate_index=index
        )
        s_population: pd.Series = get_population_series(
            self.data_dir, extrapolate_index=index
        )

        vehicle_stock_vals: list[float] = []
        for _ in range(self.bootstrap_predict_runs):
            vehicle_stock_val: float = self.bootstrap_vehicle_stock_module(
                output=self.vehicle_stock_module.vehicle_stock,
                gdp_per_capita=df_vehicle_stock.loc[year, "adjusted_gdp_per_capita"],
                run_one=True,
            )
            vehicle_stock_vals.append(vehicle_stock_val)

        df: pd.DataFrame = (
            pd.Series(vehicle_stock_vals, name="vehicle_stock")
            .quantile(self.quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .astype(np.float32)
        )
        df["adjusted_vehicle_ownership"] = df["vehicle_stock"].div(
            s_population.loc[year]
        )

        return df


@dataclasses.dataclass(kw_only=True)
class TruckStockPipeline(PerYearPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.TRUCK

    data_dir: Path
    bootstrap_fit_runs: int = 1000
    bootstrap_predict_runs: int = 1000
    quantiles: Iterable[float] = (0.025, 0.5, 0.975)

    vehicle_stock_module: TruckStockModule = dataclasses.field(
        default_factory=TruckStockModule
    )

    @functools.cached_property
    def bootstrap_vehicle_stock_module(self) -> BootstrapModule:
        df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_4_data(
            self.data_dir, vehicle=self.VEHICLE
        )

        bootstrap_vehicle_stock_module: BootstrapModule = BootstrapModule(
            module=self.vehicle_stock_module, runs=self.bootstrap_fit_runs
        )
        bootstrap_vehicle_stock_module.fit(
            log_gdp_per_capita=df_vehicle_stock["log_gdp_per_capita"].values,
            population=df_vehicle_stock["population"].values,
            vehicle_stock=df_vehicle_stock["vehicle_stock"].values,
        )
        return bootstrap_vehicle_stock_module

    def __call__(self, year: int) -> pd.DataFrame:
        index: pd.Index = pd.Index([year], name="year")

        df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_4_data(
            self.data_dir, vehicle=self.VEHICLE, extrapolate_index=index
        )
        s_population: pd.Series = get_population_series(
            self.data_dir, extrapolate_index=index
        )

        vehicle_stock_vals: list[float] = []
        for _ in range(self.bootstrap_predict_runs):
            vehicle_stock_val: float = self.bootstrap_vehicle_stock_module(
                output=self.vehicle_stock_module.vehicle_stock,
                log_gdp_per_capita=np.log(
                    df_vehicle_stock.loc[year, "adjusted_gdp_per_capita"]
                ),
                population=df_vehicle_stock.loc[year, "population"],
                run_one=True,
            )
            vehicle_stock_vals.append(vehicle_stock_val)

        df: pd.DataFrame = (
            pd.Series(vehicle_stock_vals, name="vehicle_stock")
            .quantile(self.quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .astype(np.float32)
        )
        df["adjusted_vehicle_ownership"] = df["vehicle_stock"].div(
            s_population.loc[year]
        )

        return df


@dataclasses.dataclass(kw_only=True)
class BusStockPipeline(PerYearPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.BUS

    data_dir: Path
    bootstrap_fit_runs: int = 1000
    bootstrap_predict_runs: int = 1000
    quantiles: Iterable[float] = (0.025, 0.5, 0.975)

    vehicle_stock_density_module: BusStockDensityModule = dataclasses.field(
        default_factory=BusStockDensityModule
    )

    @functools.cached_property
    def df_vehicle_stock(self) -> pd.DataFrame:
        return get_tsai_sec_2_5_data(
            self.data_dir,
            vehicle=self.VEHICLE,
            cities=set(City) - set([City.TAIWAN, City.PENGHU, City.JINMA]),
        )

    @functools.cached_property
    def min_year(self) -> int:
        return self.df_vehicle_stock["year"].min()

    @functools.cached_property
    def bootstrap_vehicle_stock_density_module(self) -> BootstrapModule:
        df_vehicle_stock = self.df_vehicle_stock

        bootstrap_vehicle_stock_density_module: BootstrapModule = BootstrapModule(
            module=self.vehicle_stock_density_module, runs=self.bootstrap_fit_runs
        )
        bootstrap_vehicle_stock_density_module.fit(
            population_density=df_vehicle_stock["population_density"].values,
            year=df_vehicle_stock["year"].values - self.min_year,
            vehicle_stock_density=df_vehicle_stock["vehicle_stock_density"].values,
        )
        return bootstrap_vehicle_stock_density_module

    def __call__(self, year: int) -> pd.DataFrame:
        index: pd.Index = pd.Index([year], name="year")

        df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_5_data(
            self.data_dir,
            vehicle=self.VEHICLE,
            cities=[City.TAIWAN],
            extrapolate_index=index,
        ).set_index("year")
        s_population: pd.Series = get_population_series(
            self.data_dir, extrapolate_index=index
        )

        vehicle_stock_density_vals: list[float] = []
        for _ in range(self.bootstrap_predict_runs):
            vehicle_stock_density_val: float = (
                self.bootstrap_vehicle_stock_density_module(
                    output=self.vehicle_stock_density_module.vehicle_stock_density,
                    population_density=df_vehicle_stock.loc[year, "population_density"],
                    year=year - self.min_year,
                    run_one=True,
                )
            )
            vehicle_stock_density_vals.append(vehicle_stock_density_val)

        df: pd.DataFrame = (
            pd.Series(vehicle_stock_density_vals, name="adjusted_vehicle_ownership")
            .quantile(self.quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .astype(np.float32)
        )
        df["vehicle_stock"] = df["adjusted_vehicle_ownership"].mul(s_population[year])

        return df


@dataclasses.dataclass(kw_only=True)
class VehicleCompositionPipeline(object):
    VEHICLE: ClassVar[Vehicle]
    FUELS: ClassVar[list[Fuel]] = [
        Fuel.INTERNAL_COMBUSTION,
        Fuel.BATTERY_ELECTRIC,
        Fuel.FULL_CELL_ELECTRIC,
    ]

    data_dir: Path
    result_dir: Path
    scenario: str = "REF"
    max_age: int = 25

    @functools.cached_property
    def s_vehicle_stock(self) -> pd.Series:
        return get_tsai_vehicle_stock_series(
            result_dir=self.result_dir, vehicle=self.VEHICLE, percentage=0.5
        )

    # this is for internal combustion vehicles as we assume all vehicles are initially internal combustion
    @functools.cached_property
    def df_vehicle_age_composition(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                Fuel.INTERNAL_COMBUSTION: get_vehicle_age_composition_series(
                    self.data_dir, vehicle=self.VEHICLE, max_age=self.max_age
                ),
                Fuel.BATTERY_ELECTRIC: pd.Series(
                    0, index=pd.RangeIndex(self.max_age + 1, name="age")
                ),
                Fuel.FULL_CELL_ELECTRIC: pd.Series(
                    0, index=pd.RangeIndex(self.max_age + 1, name="age")
                ),
            }
        )

    @functools.cached_property
    def df_vehicle_survival_rate(self) -> pd.DataFrame:
        s: pd.Series = get_tsai_vehicle_survival_rate_series(
            self.data_dir,
            result_dir=self.result_dir,
            vehicle=self.VEHICLE,
            max_age=self.max_age,
        )
        return pd.DataFrame({fuel: s for fuel in self.FUELS})

    @functools.cached_property
    def df_vehicle_market_share(self) -> pd.DataFrame:
        index: pd.Index = pd.RangeIndex(2020, 2051, name="year")

        return pd.DataFrame(
            {
                fuel: get_vehicle_market_share_series(
                    self.data_dir,
                    vehicle=self.VEHICLE,
                    fuel=fuel,
                    scenario=self.scenario,
                    extrapolate_index=index,
                )
                for fuel in self.FUELS
            }
        )

    def __call__(
        self,
        years: Iterable[int],
        # index: year
        s_vehicle_stock: pd.Series | None = None,
        # index: age, column: fuel
        df_vehicle_age_composition: pd.DataFrame | None = None,
        # index: age, column: fuel
        df_vehicle_survival_rate: pd.DataFrame | None = None,
        # index: year, column: fuel
        df_vehicle_market_share: pd.DataFrame | None = None,
    ) -> tuple[dict[int, pd.Series], dict[int, pd.DataFrame]]:
        years = list(years)

        if s_vehicle_stock is None:
            logging.info("`s_vehicle_stock` is None, using default")
            s_vehicle_stock = self.s_vehicle_stock

        if df_vehicle_age_composition is None:
            logging.info("`df_vehicle_age_composition` is None, using default")
            df_vehicle_age_composition = self.df_vehicle_age_composition

        if df_vehicle_survival_rate is None:
            logging.info("`df_vehicle_survival_rate` is None, using default")
            df_vehicle_survival_rate = self.df_vehicle_survival_rate

        if df_vehicle_market_share is None:
            logging.info("`df_vehicle_market_share` is None, using default")
            df_vehicle_market_share = self.df_vehicle_market_share

        vehicle_sale_by_year: dict[int, pd.Series] = {}
        df_vehicle_age_composition_by_year: dict[int, pd.DataFrame] = {}

        for num, year in rich.progress.track(enumerate(years), total=len(years)):
            if num == 0:
                df_vehicle_age_composition_by_year[year - 1] = (
                    s_vehicle_stock[year - 1] * df_vehicle_age_composition
                )

            df_vehicle_age_composition = df_vehicle_age_composition_by_year[year - 1]
            total_new_vehicle_sale = s_vehicle_stock[year] - s_vehicle_stock[year - 1]
            total_replacement_vehicle_sale = 0.0

            new_df_vehicle_age_composition: pd.DataFrame = pd.DataFrame()
            for fuel in self.FUELS:
                s_vehicle_survival_rate = df_vehicle_survival_rate[fuel]
                s_vehicle_age_composition = df_vehicle_age_composition[fuel]

                module = VehicleAgeCompositionModule(dims=self.max_age + 1)
                output = module(
                    age_composition=s_vehicle_age_composition.values,
                    survival_rate=s_vehicle_survival_rate.values,
                )

                new_df_vehicle_age_composition[fuel] = output["new_age_composition"]
                total_replacement_vehicle_sale += output["replacement_sale"]

            total_vehicle_sale = total_new_vehicle_sale + total_replacement_vehicle_sale

            vehicle_sale: pd.Series = pd.Series(
                {
                    fuel: df_vehicle_market_share.loc[year, fuel] * total_vehicle_sale
                    for fuel in self.FUELS
                }
            )
            new_df_vehicle_age_composition.loc[0] = vehicle_sale

            vehicle_sale_by_year[year] = vehicle_sale
            df_vehicle_age_composition_by_year[year] = new_df_vehicle_age_composition

            logging.debug(
                f"For {year} vehicle vehicle_sale is {total_vehicle_sale}, age composition is {new_df_vehicle_age_composition}"
            )

        return vehicle_sale_by_year, df_vehicle_age_composition_by_year


@dataclasses.dataclass(kw_only=True)
class CarCompositionPipeline(VehicleCompositionPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.CAR


@dataclasses.dataclass(kw_only=True)
class ScooterCompositionPipeline(VehicleCompositionPipeline):
    VEHICLE: ClassVar[Vehicle] = Vehicle.SCOOTER
