import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
from absl import app, flags

from app.data import (
    VehicleType,
    get_vehicle_ownership_data,
    get_vehicle_survival_rate_data,
)
from app.modules import (
    CarOwnershipModule,
    ScooterOwnershipModule,
    VehicleSubsidyModule,
    VehicleSurvivalRateModule,
)

flags.DEFINE_string("data_dir", "./data", "Directory for data.")
FLAGS = flags.FLAGS


def vehicle_subsidy():
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
    pprint.pprint(output_values)


def vehicle_survival_rate():
    for vehicle_type in [
        VehicleType.CAR,
        VehicleType.SCOOTER,
        VehicleType.OPERATING_CAR,
    ]:
        df: pd.DataFrame = get_vehicle_survival_rate_data(
            FLAGS.data_dir, vehicle_type=vehicle_type
        )
        module = VehicleSurvivalRateModule()
        module.fit(
            age=df.age.values, survival_rate=df.survival_rate.values, bootstrap=False
        )

        input_values: dict[str, float] = {"age": 10}
        output_values = module.forward(input_values)
        pprint.pprint(output_values)


def car_ownership():
    df: pd.DataFrame = get_vehicle_ownership_data(
        FLAGS.data_dir, vehicle_type=VehicleType.CAR
    )
    income_bin: pd.Series = (
        df.income.rank(pct=True).mul(100).astype(int).rename("income_bin")
    )
    df_agg = df.groupby(income_bin).agg(
        {"income_adjusted": np.mean, "vehicle_ownership_adjusted": np.mean}
    )

    fig = plt.Figure(figsize=(6, 4))
    (
        so.Plot(
            df_agg,
            x="income_adjusted",
            y="vehicle_ownership_adjusted",
        )
        .add(so.Dot())
        .limit(x=(0, 2e6), y=(0.1, 0.65))
        .label(x="Income", y="Vehicle Ownership")
        .on(fig)
        .plot()
    )
    fig.tight_layout()
    fig.savefig("2-2-3-car.pdf")

    module = CarOwnershipModule()
    module.fit(
        income=df_agg.income_adjusted.values,
        ownership=df_agg.vehicle_ownership_adjusted.values,
        bootstrap=False,
    )

    input_values: dict[str, float] = {
        "income": 1_000_000,
    }

    output_values = module.forward(input_values)
    pprint.pprint(output_values)


def scooter_ownership():
    df: pd.DataFrame = get_vehicle_ownership_data(
        FLAGS.data_dir, vehicle_type=VehicleType.SCOOTER
    )
    income_bin: pd.Series = (
        df.income.rank(pct=True).mul(100).astype(int).rename("income_bin")
    )
    df_agg = df.groupby(income_bin).agg(
        {"income_adjusted": np.mean, "vehicle_ownership_adjusted": np.mean}
    )

    fig = plt.Figure(figsize=(6, 4))
    (
        so.Plot(
            df_agg,
            x="income_adjusted",
            y="vehicle_ownership_adjusted",
        )
        .add(so.Dot())
        .limit(x=(0, 2e6), y=(0.3, 0.7))
        .label(x="Income", y="Vehicle Ownership")
        .on(fig)
        .plot()
    )
    fig.tight_layout()
    fig.savefig("2-2-3-scooter.pdf")

    module = ScooterOwnershipModule()
    module.fit(
        income=df_agg.income_adjusted.values,
        ownership=df_agg.vehicle_ownership_adjusted.values,
        bootstrap=False,
    )

    input_values: dict[str, float] = {
        "income": 1_000_000,
    }

    output_values = module.forward(input_values)
    pprint.pprint(output_values)


def main(_):
    # vehicle_subsidy()
    vehicle_survival_rate()
    # car_ownership()
    # scooter_ownership()


if __name__ == "__main__":
    app.run(main)
