import pprint
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
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
    get_vehicle_ownership_data,
    get_vehicle_stock_series,
    get_vehicle_survival_rate_series,
)
from app.modules import (
    CarOwnershipModule,
    IncomeDistributionModule,
    ScooterOwnershipModule,
    VehicleSubsidyModule,
    VehicleSurvivalRateModule,
)

flags.DEFINE_string("data_dir", "./data", "Directory for data.")
FLAGS = flags.FLAGS


def vehicle_subsidy():
    logging.info("Running vehicle subsidy experiment.")

    module = VehicleSubsidyModule()

    input_values: dict[str, float] = {
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

    output_values = module.forward(input_values)
    logging.info(f"Output values: {pprint.pformat(output_values)}")


def vehicle_survival_rate_experiment():
    logging.info("Running vehicle survival rate experiment.")

    objs: list[Any] = []

    for vehicle_type in [
        VehicleType.CAR,
        VehicleType.SCOOTER,
        VehicleType.OPERATING_CAR,
    ]:
        vehicle: str = vehicle_type.value.lower()

        s: pd.Series = get_vehicle_survival_rate_series(
            FLAGS.data_dir, vehicle_type=vehicle_type
        )
        objs.extend(s.reset_index().assign(vehicle=vehicle).to_dict(orient="records"))

        module = VehicleSurvivalRateModule()
        module.fit(age=s.index.values, survival_rate=s.values, bootstrap=False)

        for age in np.linspace(0, 30):
            output_values = module.forward({"age": age})

            objs.append(
                {
                    "age": age,
                    "survival_rate_fitted": output_values["survival_rate"],
                    "vehicle": vehicle,
                }
            )

    df_plot: pd.DataFrame = pd.DataFrame(objs)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    (
        so.Plot(df_plot, x="age", y="survival_rate", color="vehicle")
        .add(so.Dot())
        .on(ax)
        .plot()
    )
    (
        so.Plot(df_plot, x="age", y="survival_rate_fitted", color="vehicle")
        .add(so.Line())
        .on(ax)
        .plot()
    )
    fig.tight_layout()
    fig.savefig(f"2-2-1.pdf")


def vehicle_ownership_experiment():
    logging.info("Running vehicle ownership experiment.")

    for vehicle_type in [VehicleType.CAR, VehicleType.SCOOTER]:
        vehicle: str = vehicle_type.value.lower()

        df: pd.DataFrame = get_vehicle_ownership_data(
            FLAGS.data_dir, vehicle_type=vehicle_type, income_bins=100
        )

        if vehicle_type == VehicleType.CAR:
            ylim = (0.1, 0.65)
        elif vehicle_type == VehicleType.SCOOTER:
            ylim = (0.3, 0.7)

        fig = plt.Figure(figsize=(6, 4))
        (
            so.Plot(
                df,
                x="adjusted_income",
                y="adjusted_vehicle_ownership",
            )
            .add(so.Dot())
            .limit(x=(0, 2e6), y=ylim)
            .label(x="Income", y=f"{vehicle.title()} Ownership")
            .on(fig)
            .plot()
        )
        fig.tight_layout()
        fig.savefig(f"2-2-3-{vehicle}.pdf")

        if vehicle_type == VehicleType.CAR:
            module = CarOwnershipModule()
        else:
            module = ScooterOwnershipModule()

        module.fit(
            income=df.iloc[1:-1].adjusted_income.values,
            ownership=df.iloc[1:-1].adjusted_vehicle_ownership.values,
            bootstrap=False,
        )
        logging.info(module.param_symbol_values)


def vehicle_stock_experiment(
    income_bins_total: int = 100,
    income_bins_removed: int = 1,
    rv_expectation_samples: int = 1000,
):
    logging.info("Running vehicle stock experiment.")

    s_population: pd.Series = get_population_series(FLAGS.data_dir)
    s_vehicle_stock: pd.Series = get_vehicle_stock_series(
        FLAGS.data_dir, vehicle_type=VehicleType.CAR
    )

    df_income_distribution: pd.DataFrame = get_income_dataframe(data_dir=FLAGS.data_dir)
    df_vehicle_ownership: pd.DataFrame = get_vehicle_ownership_data(
        FLAGS.data_dir, vehicle_type=VehicleType.CAR, income_bins=income_bins_total
    )
    df_vehicle_ownership_for_fit: pd.DataFrame = df_vehicle_ownership.iloc[
        income_bins_removed:-income_bins_removed
    ]

    income_distribution_module = IncomeDistributionModule()
    car_ownership_module = CarOwnershipModule()
    car_ownership_module.fit(
        income=df_vehicle_ownership_for_fit.adjusted_income.values,
        ownership=df_vehicle_ownership_for_fit.adjusted_vehicle_ownership.values,
        bootstrap=False,
    )

    years: Iterable[int] = df_income_distribution.index.values
    for year in years:
        s_income_distribution: pd.Series = df_income_distribution.loc[year]

        stock_val: int
        outputs: dict[str, sp.Basic]
        if year in s_vehicle_stock.index:
            stock_val = int(s_vehicle_stock[year])

        else:
            outputs = income_distribution_module.forward(
                {
                    "income": s_income_distribution.adjusted_income,
                    "gini": s_income_distribution.gini,
                }
            )
            income_rv: sp.Basic = outputs["income_rv"]

            outputs = car_ownership_module.forward({"income": income_rv})
            ownership_rv: sp.Basic = outputs["ownership"]

            ownership_val = sps.E(ownership_rv, numsamples=rv_expectation_samples)
            stock_val = int(ownership_val * s_population[year])

        logging.info(f"Total stock of cars in {year}: {stock_val}")
        logging.info(s_income_distribution.to_dict())


def main(_):
    logging.set_verbosity(logging.INFO)

    vehicle_subsidy()
    vehicle_survival_rate_experiment()
    vehicle_ownership_experiment()
    vehicle_stock_experiment()


if __name__ == "__main__":
    app.run(main)
