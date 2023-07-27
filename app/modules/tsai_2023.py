# Tsai, Chia-Yu, Tsung-Heng Chang, and I-Yun Lisa Hsieh.
#   "Evaluating vehicle fleet electrification against net-zero targets in scooter-dominated road transport."
#   Transportation Research Part D: Transport and Environment 114 (2023): 103542.
#
# https://www.sciencedirect.com/science/article/pii/S1361920922003686


from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.optimize
import sklearn.metrics
import sympy as sp
import sympy.stats as sps
from absl import logging

from app.modules.base import BaseModule, Module
from app.modules.core import (
    GammaCurveModule,
    GompertzCurveModule,
    LinearModule,
    SigmoidCurveModule,
)


# Section 2.2.1: Stock and Sales
class VehicleSurvivalRateModule(Module):
    def __init__(self):
        age = sp.Symbol("age")
        a = sp.Symbol("a")
        b = sp.Symbol("b")

        survival_rate = 1 / (1 + a * sp.exp(b * age))

        self.age = age
        self.a = a
        self.b = b

        self.survival_rate = survival_rate

    def output(self) -> dict[str, sp.Basic]:
        return {
            "a": self.a,
            "b": self.b,
            "survival_rate": self.survival_rate,
        }

    def _fit(self, age: np.ndarray, survival_rate: np.ndarray) -> dict[sp.Symbol, float]:  # type: ignore
        (a, b), _ = scipy.optimize.curve_fit(
            sp.lambdify([self.age, self.a, self.b], self.survival_rate),
            age,
            survival_rate,
            p0=[0.005, 0.5],
        )
        return {self.a: a, self.b: b}


# Section 2.2.2: Disposable Income Distribution
class IncomeDistributionModule(BaseModule):
    def __init__(self):
        mean_income = sp.Symbol("mean_income")
        gini = sp.Symbol("gini")

        beta = 1 / gini
        alpha = mean_income * sp.sin(sp.pi / beta) / (sp.pi / beta)

        income_var = sp.Symbol("income_var")
        income_pdf = (
            (beta / alpha)
            * (income_var / alpha) ** (beta - 1)
            / (1 + (income_var / alpha) ** beta) ** 2
        )
        income_rv = sps.ContinuousRV(income_var, income_pdf, set=sp.Interval(0, sp.oo))

        self.mean_income = mean_income
        self.gini = gini

        self.beta = beta
        self.alpha = alpha
        self.income_var = income_var
        self.income_pdf = income_pdf
        self.income_rv = income_rv

    def output(self) -> dict[str, sp.Basic]:
        return {
            "beta": self.beta,
            "alpha": self.alpha,
            "income_pdf": self.income_pdf,
            "income_rv": self.income_rv,
        }


# Section 2.2.3: Ownership Probability Function
class CarOwnershipModule(GompertzCurveModule):
    def __init__(self):
        super().__init__()

        self.income = self.x
        self.ownership = self.y

    def output(self) -> dict[str, sp.Basic]:
        return {"ownership": self.ownership, **super().output()}

    def __call__(self, output: Any = None, **inputs: sp.Basic) -> Any:
        income: sp.Float = cast(sp.Float, inputs.pop("income"))
        income_in_millions: sp.Float = income / 1_000_000

        return super().__call__(output, income=income_in_millions, **inputs)

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=income_in_millions,
            y=ownership,
            p0=[np.max(ownership), -2, -2],
        )

        return params


class CarOwnershipModuleV2(SigmoidCurveModule):
    def __init__(self):
        super().__init__()

        self.income = self.x
        self.ownership = self.y

    def output(self) -> dict[str, sp.Basic]:
        return {"ownership": self.ownership, **super().output()}

    def __call__(self, output: Any = None, **inputs: sp.Basic) -> Any:
        income: sp.Float = cast(sp.Float, inputs.pop("income"))
        income_in_millions: sp.Float = income / 1_000_000

        return super().__call__(output, income=income_in_millions, **inputs)

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=income_in_millions,
            y=ownership,
            p0=[
                np.max(ownership),
                2 * (1 - np.min(ownership) / np.max(ownership)),
                -1.0,
            ],
        )

        return params


class ScooterOwnershipModule(GammaCurveModule):
    def __init__(self):
        super().__init__()

        self.income = self.x
        self.ownership = self.y

    def output(self) -> dict[str, sp.Basic]:
        return {"ownership": self.ownership} | super().output()

    def __call__(self, output: Any = None, **inputs: sp.Basic) -> Any:
        income: sp.Float = cast(sp.Float, inputs.pop("income"))
        income_in_millions: sp.Float = income / 1_000_000

        return super().__call__(output, income=income_in_millions, **inputs)

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=income_in_millions,
            y=ownership,
            p0=[1.5, 1, 0.1],
        )

        return params


# Section 2.3: Non-private Cars Module
class OperatingCarStockModule(GompertzCurveModule):
    def __init__(self):
        super().__init__()

        self.gdp_per_capita = self.x
        self.vehicle_stock = self.y

    def output(self) -> dict[str, sp.Basic]:
        return {"vehicle_stock": self.vehicle_stock} | super().output()

    def _fit(  # type: ignore
        self, gdp_per_capita: np.ndarray, vehicle_stock: np.ndarray
    ) -> dict[sp.Basic, float]:
        gdp_per_capita_in_millions: np.ndarray = gdp_per_capita / 1_000_000
        vehicle_stock_in_millions: np.ndarray = vehicle_stock / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=gdp_per_capita_in_millions,
            y=vehicle_stock_in_millions,
            p0=[np.max(gdp_per_capita_in_millions), -80, -8],
        )
        params[self.beta] /= 1_000_000
        params[self.gamma] *= 1_000_000

        return params


# Section 2.4: Freight Vehicles Module
class TruckStockModule(LinearModule):
    def __init__(self):
        super().__init__(input_dims=2)

        self.log_gdp_per_capita = self.x[0]
        self.population = self.x[1]
        self.vehicle_stock = self.y

    def output(self) -> dict[str, sp.Basic]:
        return {"vehicle_stock": self.y} | super().output()

    def _fit(  # type: ignore
        self,
        log_gdp_per_capita: np.ndarray,
        population: np.ndarray,
        vehicle_stock: np.ndarray,
    ) -> dict[sp.Basic, float]:
        params: dict[sp.Basic, float] = super()._fit(
            X=np.r_["1,2,0", log_gdp_per_capita, population],
            y=vehicle_stock,
        )

        return params


# Section 2.5: Bus Module
class BusStockModule(Module):
    def __init__(self):
        self.sigmoid_curve = SigmoidCurveModule()
        self.linear = LinearModule(bias=False)

        self.population_density = self.sigmoid_curve.x
        self.year = self.linear.x[0]

        self.vehicle_stock_density = self.sigmoid_curve.y + self.linear.y

    def __call__(self, output: Any = None, **inputs: sp.Basic) -> Any:
        population_density: sp.Float = cast(sp.Float, inputs.pop("population_density"))
        population_density_in_thousands: sp.Float = population_density / 1_000

        return super().__call__(
            output, population_density=population_density_in_thousands, **inputs
        )

    def output(self) -> dict[str, sp.Basic]:
        return {
            "vehicle_stock_density": self.vehicle_stock_density,
        }

    def _fit(  # type: ignore
        self,
        population_density: np.ndarray,
        year: np.ndarray,
        vehicle_stock_density: np.ndarray,
    ) -> dict[sp.Basic, float]:
        population_density_in_thousands: np.ndarray = population_density / 1_000

        fn: Callable[[tuple, float, float, float, float], np.ndarray] = sp.lambdify(
            [
                (self.population_density, self.year),
                self.sigmoid_curve.gamma,
                self.sigmoid_curve.alpha,
                self.sigmoid_curve.beta,
                self.linear.a[0],
            ],
            self.vehicle_stock_density,
        )
        (gamma, alpha, beta, a_0), _ = scipy.optimize.curve_fit(
            fn,
            (population_density_in_thousands, year),
            vehicle_stock_density,
            p0=[2.5 / 1_000, 1.5, -0.5, 0.02 / 1_000],
        )

        y: np.ndarray = vehicle_stock_density
        y_pred = np.asarray(
            [
                fn((p, y), gamma, alpha, beta, a_0)
                for p, y in zip(population_density_in_thousands, year)
            ]
        )
        r2: float = sklearn.metrics.r2_score(y, y_pred)

        logging.debug(
            f"Fitted with r2={r2} and parameters: gamma={gamma}, alpha={alpha}, beta={beta}, a_0={a_0}"
        )

        return {
            self.sigmoid_curve.gamma: gamma,
            self.sigmoid_curve.alpha: alpha,
            self.sigmoid_curve.beta: beta,
            self.linear.a[0]: a_0,
        }
