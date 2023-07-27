from pathlib import Path

import numpy as np

from main import (
    tsai_2023_sec_2_2_1_experiment,
    tsai_2023_sec_2_2_2_experiment,
    tsai_2023_sec_2_2_3_experiment,
    tsai_2023_sec_2_3_experiment,
    tsai_2023_sec_2_4_experiment,
    tsai_2023_sec_2_5_experiment,
    tsai_2023_sec_3_1_experiment,
    vehicle_subsidy,
)


def test_vehicle_subsidy(data_dir: Path, result_dir: Path):
    vehicle_subsidy(data_dir, result_dir)


def test_tsai_2023_sec_2_2_1_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_2_1_experiment(
        data_dir, result_dir, plot_age_values=np.linspace(0, 30, 2)
    )


def test_tsai_2023_sec_2_2_2_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_2_2_experiment(
        data_dir,
        result_dir,
        plot_years=range(2000, 2010, 10),
        plot_year_colors=["b"],
        plot_income_values=np.linspace(0, 1_000_000, 2),
    )


def test_tsai_2023_sec_2_2_3_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_2_3_experiment(
        data_dir,
        result_dir,
        bootstrap_runs=1,
        plot_income_values=np.linspace(0, 2_000_000, 2),
        plot_ownership_quantiles=np.arange(0, 1.101, 1.0),
    )


def test_tsai_2023_sec_2_3_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_3_experiment(
        data_dir,
        result_dir,
        bootstrap_runs=1,
        plot_gdp_per_capita_values=np.linspace(600_000, 1_500_000, 2),
        plot_stock_quantiles=np.arange(0, 1.001, 1.0),
    )


def test_tsai_2023_sec_2_4_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_4_experiment(
        data_dir,
        result_dir,
        bootstrap_runs=1,
        plot_stock_quantiles=np.arange(0, 1.001, 1.0),
    )


def test_tsai_2023_sec_2_5_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_2_5_experiment(
        data_dir,
        result_dir,
        bootstrap_runs=1,
        plot_population_density_values=np.linspace(0, 10_000, 2),
        plot_years=np.arange(1998, 1999),
        plot_stock_quantiles=np.arange(0, 1.001, 1.0),
    )


def test_tsai_2023_sec_3_1_experiment(data_dir: Path, result_dir: Path):
    tsai_2023_sec_3_1_experiment(
        data_dir,
        result_dir,
        bootstrap_runs=1,
        integrate_sigma=1,
        predict_years=np.arange(2020, 2021),
        plot_years=np.arange(2000, 2021),
    )
