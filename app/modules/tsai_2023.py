# Tsai, Chia-Yu, Tsung-Heng Chang, and I-Yun Lisa Hsieh.
#   "Evaluating vehicle fleet electrification against net-zero targets in scooter-dominated road transport."
#   Transportation Research Part D: Transport and Environment 114 (2023): 103542.
#
# https://www.sciencedirect.com/science/article/pii/S1361920922003686


import numpy as np
import scipy.optimize
import sympy as sp
import sympy.stats as sps
import sympy.stats.rv

from app.modules.base import FittableModule, Module
from app.modules.core import GammaDistributionModule, GompertzDistributionModule


# Section 2.2.1: Stock and Sales
class VehicleSurvivalRateModule(FittableModule):
    def __init__(self):
        age = sp.Symbol("age")
        a = sp.Symbol("a")
        b = sp.Symbol("b")

        survival_rate = 1 / (1 + a * sp.exp(b * age))

        self.age = age
        self.a = a
        self.b = b

        self.survival_rate = survival_rate

    def output_symbols(self) -> dict[str, sp.Basic]:
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
class IncomeDistributionModule(Module):
    def __init__(self):
        income = sp.Symbol("income")
        gini = sp.Symbol("gini")

        beta = 1 / gini
        alpha = income * sp.sin(sp.pi / beta) / (sp.pi / beta)

        income_var = sp.Symbol("income_var")
        income_pdf = (
            (beta / alpha)
            * (income_var / alpha) ** (beta - 1)
            / (1 + (income_var / alpha) ** beta) ** 2
        )
        income_rv = sps.ContinuousRV(income_var, income_pdf, set=sp.Interval(0, sp.oo))

        self.income = income
        self.gini = gini

        self.beta = beta
        self.alpha = alpha
        self.income_pdf = income_pdf
        self.income_rv = income_rv

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "beta": self.beta,
            "alpha": self.alpha,
            "income_pdf": self.income_pdf,
            "income_rv": self.income_rv,
        }


# Section 2.2.3: Ownership Probability Function
class CarOwnershipModule(GompertzDistributionModule):
    def __init__(self):
        super().__init__()

        self.income = self.x
        self.ownership = self.y

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {"ownership": self.y, **super().output_symbols()}

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=income_in_millions, y=ownership, p0=[np.max(ownership), -2, -2]
        )
        params[self.beta] /= 1_000_000

        return params


class ScooterOwnershipModule(GammaDistributionModule):
    def __init__(self):
        super().__init__()

        self.income = self.x
        self.ownership = self.y

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {"ownership": self.y, **super().output_symbols()}

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=income_in_millions, y=ownership, p0=[1.5, 1, 0.1]
        )
        params[self.beta] /= 1_000_000

        return params


# Section 2.3: Non-private Cars Module
class OperatingCarStockModule(GompertzDistributionModule):
    def __init__(self):
        super().__init__()

        self.gdp_per_capita = self.x
        self.stock = self.y

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {"stock": self.y, **super().output_symbols()}

    def _fit(self, gdp_per_capita: np.ndarray, stock: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        gdp_per_capita_in_millions: np.ndarray = gdp_per_capita / 1_000_000
        stock_in_millions: np.ndarray = stock / 1_000_000

        params: dict[sp.Basic, float] = super()._fit(
            x=gdp_per_capita_in_millions,
            y=stock_in_millions,
            p0=[np.max(gdp_per_capita_in_millions), -80, -8],
        )
        params[self.beta] /= 1_000_000
        params[self.gamma] *= 1_000_000

        return params
