import itertools
import logging as py_logging
import pprint
from collections.abc import Iterable
from typing import Any

import graphviz as gv
import numpy as np
import pandas as pd
import seaborn.objects as so
import sympy as sp
import sympy.stats as sps
from absl import app, flags, logging

from app.data import (
    VehicleType,
    get_income_dataframe,
    get_population_series,
    get_tsai_sec_2_2_3_data,
    get_tsai_sec_2_3_data,
    get_tsai_sec_2_4_data,
    get_tsai_sec_2_5_data,
    get_vehicle_stock_series,
    get_vehicle_survival_rate_series,
)
from app.modules import (
    BaseModule,
    BusStockModule,
    CarOwnershipModule,
    IncomeDistributionModule,
    OperatingCarStockModule,
    ScooterOwnershipModule,
    TruckStockModule,
    VehicleSubsidyModule,
    VehicleSurvivalRateModule,
)

flags.DEFINE_string("data_dir", "./data", "Directory for data.")
FLAGS = flags.FLAGS


def vehicle_subsidy():
    logging.info("Running vehicle subsidy experiment.")

    module = VehicleSubsidyModule()

    inputs: dict[str, float] = {
        "d": 14_332,  # km/year
        "f_e": 0.15,  # kWh/km
        "f_f": 0.08,  # L/km
        "ρ_e": 0.081,  # $/kWh
        "ρ_f": 0.997,  # $/L
        "M_e": 743,  # $/year
        "M_f": 694,  # $/year
        "e": 0.14,  # kg/km
        "Q": 2_000,  # kg
        "T": 10,  # year
        "F_e": 6,  # h
        "F_f": 0.0833,  # h
        "I_e": 10,
        "C": 25_000,  # $
        "k": 1.16,
        "i_e": 0,
        "i_f": 230.75,  # $/year
        "ε": 0.1,
        "θ": 0.69,
        "β_1": 1.211e-5,
        "β_2": 0.05555,
        "β_3": 0.01831,
        "λ_1": 0.5,
        "λ_2": 0.5,
        "ΔN_v": 100_000,
    }

    output = module(**inputs)
    logging.info(f"Output values: {pprint.pformat(output)}")

    dot_str: str = sp.dotprint(
        module.E_G,
        atom=lambda x: not isinstance(x, sp.Basic),
    )
    gv.Source(dot_str).render("total_budget.gv", format="png")


def tsai_2023_sec_2_2_1_experiment():
    logging.info("Running Tsai 2023 Section 2.2.1 experiment.")

    objs: list[dict[str, Any]] = []

    for vehicle_type in [
        VehicleType.CAR,
        VehicleType.SCOOTER,
        VehicleType.OPERATING_CAR,
    ]:
        vehicle: str = vehicle_type.value.lower()

        s: pd.Series = get_vehicle_survival_rate_series(
            FLAGS.data_dir, vehicle_type=vehicle_type
        )
        objs.extend(
            s.reset_index().assign(vehicle=vehicle, zorder=1).to_dict(orient="records")
        )

        module = VehicleSurvivalRateModule()
        module.fit(age=s.index.values, survival_rate=s.values, bootstrap=False)

        for age in np.linspace(0, 30):
            output = module(age=age)

            objs.append(
                {
                    "zorder": 0,
                    "age": age,
                    "survival_rate": output["survival_rate"],
                    "vehicle": vehicle,
                }
            )

    df_plot: pd.DataFrame = pd.DataFrame(objs)

    (
        so.Plot(
            df_plot,
            x="age",
            y="survival_rate",
            color="vehicle",
            marker="zorder",
            linewidth="zorder",
            linestyle="zorder",
        )
        .add(so.Line())
        .scale(
            color={"car": "b", "scooter": "r", "operating_car": "g"},
            marker={0: None, 1: "o"},
            linewidth={0: 2, 1: 0},
            linestyle={0: (6, 2), 1: "-"},
        )
        .label(x="Age (year)", y="Survival Rate")
        .layout(size=(6, 4))
        .save("tsai-2023-sec-2-2-1.pdf")
    )


def tsai_2023_sec_2_2_2_experiment(
    plot_years: Iterable[int] = range(2000, 2060, 10),
    plot_year_colors: Iterable[str] = [
        "b",
        "tab:orange",
        "g",
        "r",
        "tab:purple",
        "tab:brown",
    ],
    plot_income_values: Iterable[float] = np.linspace(0, 1_000_000, 1_000),
):
    logging.info("Running Tsai 2023 Section 2.2.2 experiment.")

    df_income_distribution: pd.DataFrame = get_income_dataframe(data_dir=FLAGS.data_dir)
    income_distribution_module = IncomeDistributionModule()

    plot_objs: list[Any] = []
    for year in plot_years:
        logging.info(f"Running year {year}.")

        s_income_distribution: pd.Series = df_income_distribution.loc[year]

        output = income_distribution_module(
            income=s_income_distribution.adjusted_income,
            gini=s_income_distribution.gini,
        )
        income_rv: sp.Basic = output["income_rv"]

        for income in plot_income_values:
            income_pdf: float = sps.density(income_rv)(income).evalf()
            plot_objs.append(
                {
                    "year": year,
                    "income": income,
                    "income_pdf": income_pdf,
                }
            )

    df_plot: pd.DataFrame = pd.DataFrame(plot_objs)
    (
        so.Plot(
            df_plot,
            x="income",
            y="income_pdf",
            color="year",
        )
        .add(so.Line())
        .scale(color=dict(zip(plot_years, plot_year_colors)))
        .label(x="Disposable Income", y="Probability Density")
        .layout(size=(6, 4))
        .save("tsai-2023-sec-2-2-2.pdf")
    )


def tsai_2023_sec_2_2_3_experiment(
    income_bins_total: int = 100,
    income_bins_removed: int = 1,
    bootstrap_runs: int = 100,
    plot_income_values: Iterable[float] = np.linspace(0, 2_000_000, 1_000),
    plot_ownership_quantiles: Iterable[float] = np.arange(0, 1.001, 0.1),
):
    logging.info("Running Tsai 2023 Section 2.2.3 experiment.")

    for vehicle_type in [VehicleType.CAR, VehicleType.SCOOTER]:
        module: BaseModule

        if vehicle_type == VehicleType.CAR:
            module = CarOwnershipModule()
        elif vehicle_type == VehicleType.SCOOTER:
            module = ScooterOwnershipModule()
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}.")

        vehicle: str = vehicle_type.value.lower()
        vehicle_title: str = vehicle_type.replace("_", " ").title()
        logging.info(f"Vehicle type: {vehicle}")

        df_vehicle_ownership: pd.DataFrame = get_tsai_sec_2_2_3_data(
            FLAGS.data_dir, vehicle_type=vehicle_type, income_bins=income_bins_total
        )
        df_vehicle_ownership_to_fit: pd.DataFrame = df_vehicle_ownership.iloc[
            income_bins_removed:-income_bins_removed
        ]

        plot_objs: list[dict[str, Any]] = (
            df_vehicle_ownership.reset_index()
            .assign(percentage=-1, zorder=1)
            .to_dict(orient="records")
        )

        curve_objs: list[dict[str, Any]] = []
        for run in range(bootstrap_runs):
            logging.info(f"Running bootstrap vehicle_type={vehicle_type}, run={run}.")

            module.fit(
                income=df_vehicle_ownership_to_fit.adjusted_income.values,
                ownership=df_vehicle_ownership_to_fit.adjusted_vehicle_ownership.values,
                bootstrap=True,
            )

            for income in plot_income_values:
                output = module(income=income)

                curve_objs.append(
                    {
                        "adjusted_income": income,
                        "adjusted_vehicle_ownership": float(output["ownership"]),
                    }
                )

        plot_objs.extend(
            pd.DataFrame(curve_objs)
            .groupby("adjusted_income")
            .quantile(plot_ownership_quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .assign(zorder=0)
            .to_dict(orient="records")
        )

        df_plot: pd.DataFrame = pd.DataFrame(plot_objs)
        (
            so.Plot(
                df_plot,
                x="adjusted_income",
                y="adjusted_vehicle_ownership",
                color="zorder",
                marker="zorder",
                linewidth="zorder",
                linestyle="zorder",
            )
            .add(so.Line(), group="percentage")
            .scale(
                color={0: "gray", 1: "b"},
                marker={0: None, 1: "o"},
                linewidth={0: 2, 1: 0},
                linestyle={0: (6, 2), 1: "-"},
            )
            .limit(
                x=(np.min(plot_income_values), np.max(plot_income_values)),
                y=(0, 0.8),
            )
            .label(x="Disposable Income", y=f"{vehicle_title} Ownership")
            .layout(size=(6, 4))
            .save(f"tsai-2023-sec-2-2-3-{vehicle}.pdf")
        )


def tsai_2023_sec_2_3_experiment(
    bootstrap_runs: int = 100,
    plot_gdp_per_capita_values: Iterable[float] = np.linspace(600_000, 1_500_000, 100),
    plot_stock_quantiles: Iterable[float] = np.arange(0, 1.001, 0.1),
):
    logging.info("Running Tsai 2023 Section 2.3 experiment.")

    vehicle_type: VehicleType = VehicleType.OPERATING_CAR
    vehicle_title: str = vehicle_type.replace("_", " ").title()

    df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_3_data(
        FLAGS.data_dir, vehicle_type=vehicle_type
    )
    plot_objs: list[dict[str, Any]] = (
        df_vehicle_stock.reset_index()
        .assign(percentage=-1, zorder=1)
        .to_dict(orient="records")
    )

    module = OperatingCarStockModule()

    curve_objs: list[dict[str, Any]] = []
    for run in range(bootstrap_runs):
        logging.info(f"Running bootstrap vehicle_type={vehicle_type}, run={run}.")

        module.fit(
            gdp_per_capita=df_vehicle_stock.adjusted_gdp_per_capita.values,
            vehicle_stock=df_vehicle_stock.vehicle_stock.values,
            bootstrap=True,
        )

        for gdp_per_capita in plot_gdp_per_capita_values:
            output = module(gdp_per_capita=gdp_per_capita)

            curve_objs.append(
                {
                    "adjusted_gdp_per_capita": gdp_per_capita,
                    "vehicle_stock": float(output["vehicle_stock"]),
                }
            )

    plot_objs.extend(
        pd.DataFrame(curve_objs)
        .groupby("adjusted_gdp_per_capita")
        .quantile(plot_stock_quantiles)
        .rename_axis(index={None: "percentage"})
        .reset_index()
        .assign(zorder=0)
        .to_dict(orient="records")
    )

    df_plot: pd.DataFrame = pd.DataFrame(plot_objs)
    (
        so.Plot(
            df_plot,
            x="adjusted_gdp_per_capita",
            y="vehicle_stock",
            color="zorder",
            marker="zorder",
            linewidth="zorder",
            linestyle="zorder",
        )
        .add(so.Line(), group="percentage")
        .scale(
            color={0: "gray", 1: "b"},
            marker={0: None, 1: "o"},
            linewidth={0: 2, 1: 0},
            linestyle={0: (6, 2), 1: "-"},
        )
        .label(x="GDP per Capita", y=f"{vehicle_title} Stock")
        .layout(size=(6, 4))
        .save("tsai-2023-sec-2-3.pdf")
    )


def tsai_2023_sec_2_4_experiment(
    bootstrap_runs: int = 100,
    plot_stock_quantiles: Iterable[float] = np.arange(0, 1.001, 0.1),
):
    logging.info("Running Tsai 2023 Section 2.4 experiment.")

    vehicle_type: VehicleType = VehicleType.TRUCK
    vehicle_title: str = vehicle_type.replace("_", " ").title()

    df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_4_data(
        FLAGS.data_dir, vehicle_type=vehicle_type
    )

    module = TruckStockModule()

    curve_objs: list[dict[str, Any]] = []
    for run in range(bootstrap_runs):
        logging.info(f"Running bootstrap vehicle_type={vehicle_type}, run={run}.")

        module.fit(
            log_gdp_per_capita=df_vehicle_stock.log_gdp_per_capita.values,
            population=df_vehicle_stock.population.values,
            vehicle_stock=df_vehicle_stock.vehicle_stock.values,
            bootstrap=True,
        )

        for _, s_vehicle_stock in df_vehicle_stock.iterrows():
            output = module(
                log_gdp_per_capita=s_vehicle_stock.log_gdp_per_capita,
                population=s_vehicle_stock.population,
            )

            curve_objs.append(
                {
                    "log_gdp_per_capita": s_vehicle_stock.log_gdp_per_capita,
                    "population": s_vehicle_stock.population,
                    "vehicle_stock": float(output["vehicle_stock"]),
                }
            )

    plot_objs: list[dict[str, Any]]

    for plot_against in ["log_gdp_per_capita", "population"]:
        name: str
        xlabel: str
        if plot_against == "log_gdp_per_capita":
            name = "gdp"
            xlabel = "Logarithm of GDP per Capita"
        elif plot_against == "population":
            name = "population"
            xlabel = "Population"

        plot_objs = (
            df_vehicle_stock.reset_index()
            .assign(percentage=-1, zorder=1)
            .to_dict(orient="records")
        )
        plot_objs.extend(
            pd.DataFrame(curve_objs)
            .groupby(plot_against)
            .quantile(plot_stock_quantiles)
            .rename_axis(index={None: "percentage"})
            .reset_index()
            .assign(zorder=0)
            .to_dict(orient="records")
        )

        df_plot: pd.DataFrame = pd.DataFrame(plot_objs)
        (
            so.Plot(
                df_plot,
                x=plot_against,
                y="vehicle_stock",
                color="zorder",
                marker="zorder",
                linewidth="zorder",
                linestyle="zorder",
            )
            .add(so.Line(), group="percentage")
            .scale(
                color={0: "gray", 1: "b"},
                marker={0: None, 1: "o"},
                linewidth={0: 2, 1: 0},
                linestyle={0: (6, 2), 1: "-"},
            )
            .label(x=xlabel, y=f"{vehicle_title} Stock")
            .layout(size=(6, 4))
            .save(f"tsai-2023-sec-2-4-{name}.pdf")
        )


def tsai_2023_sec_2_5_experiment(
    bootstrap_runs: int = 100,
    plot_population_density_values: Iterable[float] = np.linspace(0, 10_000, 100),
    plot_years: Iterable[int] = np.arange(1998, 2023),
    plot_stock_quantiles: Iterable[float] = np.arange(0, 1.001, 0.1),
):
    logging.info("Running Tsai 2023 Section 2.5 experiment.")

    vehicle_type: VehicleType = VehicleType.BUS
    vehicle_title: str = vehicle_type.replace("_", " ").title()

    df_vehicle_stock: pd.DataFrame = get_tsai_sec_2_5_data(
        data_dir=FLAGS.data_dir, vehicle_type=vehicle_type
    )
    min_year: int = df_vehicle_stock.year.min()

    plot_objs: list[dict[str, Any]] = (
        df_vehicle_stock.reset_index()
        .assign(percentage=-1, zorder=1)
        .to_dict(orient="records")
    )

    module = BusStockModule()

    curve_objs: list[dict[str, Any]] = []
    for run in range(bootstrap_runs):
        logging.info(f"Running bootstrap vehicle_type={vehicle_type}, run={run}.")

        module.fit(
            population_density=df_vehicle_stock.population_density.values,
            year=df_vehicle_stock.year.values - min_year,
            vehicle_stock_density=df_vehicle_stock.vehicle_stock_density.values,
            bootstrap=True,
        )

        for population_density, year in itertools.product(
            plot_population_density_values, plot_years
        ):
            output = module(population_density=population_density, year=year - min_year)

            curve_objs.append(
                {
                    "population_density": population_density,
                    "year": year,
                    "vehicle_stock_density": float(output["vehicle_stock_density"]),
                }
            )

    plot_objs.extend(
        pd.DataFrame(curve_objs)
        .groupby("population_density")
        .quantile(plot_stock_quantiles)
        .rename_axis(index={None: "percentage"})
        .reset_index()
        .assign(zorder=0)
        .to_dict(orient="records")
    )

    df_plot: pd.DataFrame = pd.DataFrame(plot_objs)
    (
        so.Plot(
            df_plot,
            x="population_density",
            y="vehicle_stock_density",
            color="zorder",
            marker="zorder",
            linewidth="zorder",
            linestyle="zorder",
        )
        .add(so.Line(), group="percentage")
        .scale(
            color={0: "gray", 1: "b"},
            marker={0: None, 1: "o"},
            linewidth={0: 2, 1: 0},
            linestyle={0: (6, 2), 1: "-"},
        )
        .label(x="Population Density", y=f"{vehicle_title} Stock Density")
        .layout(size=(6, 4))
        .save(f"tsai-2023-sec-2-5.pdf")
    )


def tsai_2023_sec_3_1_experiment(
    income_bins_total: int = 100,
    income_bins_removed: int = 1,
    rv_expectation_samples: int = 1000,
):
    logging.info("Running Tsai 2023 Section 3.1 experiment.")

    s_population: pd.Series = get_population_series(FLAGS.data_dir)
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        FLAGS.data_dir, vehicle_type=VehicleType.CAR
    )

    df_income_distribution: pd.DataFrame = get_income_dataframe(data_dir=FLAGS.data_dir)
    df_vehicle_ownership: pd.DataFrame = get_tsai_sec_2_2_3_data(
        FLAGS.data_dir, vehicle_type=VehicleType.CAR, income_bins=income_bins_total
    )
    df_vehicle_ownership_to_fit: pd.DataFrame = df_vehicle_ownership.iloc[
        income_bins_removed:-income_bins_removed
    ]

    income_distribution_module = IncomeDistributionModule()
    car_ownership_module = CarOwnershipModule()
    car_ownership_module.fit(
        income=df_vehicle_ownership_to_fit.adjusted_income.values,
        ownership=df_vehicle_ownership_to_fit.adjusted_vehicle_ownership.values,
        bootstrap=False,
    )

    years: Iterable[int] = df_income_distribution.index.values
    for year in years:
        s_income_distribution: pd.Series = df_income_distribution.loc[year]

        stock_val: int
        output: dict[str, sp.Basic]
        if year in s_vehicle_stock.index:
            stock_val = int(s_vehicle_stock[year])

        else:
            output = income_distribution_module(
                income=s_income_distribution.adjusted_income,
                gini=s_income_distribution.gini,
            )
            income_rv: sp.Basic = output["income_rv"]

            output = car_ownership_module(income=income_rv)
            ownership_rv: sp.Basic = output["ownership"]

            ownership_val = sps.E(ownership_rv, numsamples=rv_expectation_samples)
            stock_val = int(ownership_val * s_population[year])

        logging.info(f"Total stock of cars in {year}: {stock_val}")
        logging.info(s_income_distribution.to_dict())


def main(_):
    py_logging.getLogger("matplotlib.category").setLevel(py_logging.WARNING)

    logging.set_verbosity(logging.DEBUG)

    vehicle_subsidy()
    tsai_2023_sec_2_2_1_experiment()
    tsai_2023_sec_2_2_2_experiment()
    tsai_2023_sec_2_2_3_experiment()
    tsai_2023_sec_2_3_experiment()
    tsai_2023_sec_2_4_experiment()
    tsai_2023_sec_2_5_experiment()
    tsai_2023_sec_3_1_experiment()


if __name__ == "__main__":
    app.run(main)
