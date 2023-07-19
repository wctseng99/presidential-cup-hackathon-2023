# Tsai, Chia-Yu, Tsung-Heng Chang, and I-Yun Lisa Hsieh.
#   "Evaluating vehicle fleet electrification against net-zero targets in scooter-dominated road transport."
#   Transportation Research Part D: Transport and Environment 114 (2023): 103542.
#
# https://www.sciencedirect.com/science/article/pii/S1361920922003686


import numpy as np
import scipy.optimize
import sympy as sp
import sympy.stats

from app.modules.base import FittableModule, Module


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
        return {"a": a, "b": b}


# Section 2.2.2: Disposable Income Distribution
class DisposableIncomeDistributionModule(Module):
    def __init__(self):
        income = sp.Symbol("income")
        gini_coefficient = sp.Symbol("gini_coefficient")

        beta = 1 / gini_coefficient
        alpha = income * sp.sin(sp.pi / beta) / (sp.pi / beta)

        x = sp.Symbol("x")
        income_rv = sympy.stats.ContinuousRV(
            x,
            (beta / alpha) * (x / alpha) ** (beta - 1) / (1 + (x / alpha) ** beta) ** 2,
            set=sp.Interval(0, sp.oo),
        )

        self.income = income
        self.gini_coefficient = gini_coefficient

        self.beta = beta
        self.alpha = alpha
        self.income_rv = income_rv

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "income": self.income,
            "beta": self.beta,
            "alpha": self.alpha,
            "income_rv": self.income_rv,
        }


# Section 2.2.3: Ownership Probability Function
class CarOwnershipModule(FittableModule):
    def __init__(self):
        income = sp.Symbol("income")
        gamma = sp.Symbol("gamma")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")

        ownership = gamma * sp.exp(alpha * sp.exp(beta * income))

        self.income = income
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.ownership = ownership

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "ownership": self.ownership,
        }

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        (gamma, alpha, beta), _ = scipy.optimize.curve_fit(
            sp.lambdify(
                [self.income, self.gamma, self.alpha, self.beta], self.ownership
            ),
            income_in_millions,
            ownership,
            p0=[np.max(ownership), -2, -2],
        )
        return {self.gamma: gamma, self.alpha: alpha, self.beta: beta / 1_000_000}


class ScooterOwnershipModule(FittableModule):
    def __init__(self):
        income = sp.Symbol("income")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")
        C = sp.Symbol("C")

        ownership = (
            (beta**alpha) * (income ** (alpha - 1)) * sp.exp(-beta * income)
        ) / sp.gamma(alpha) + C

        self.income = income
        self.alpha = alpha
        self.beta = beta
        self.C = C

        self.ownership = ownership

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "C": self.C,
            "ownership": self.ownership,
        }

    def _fit(self, income: np.ndarray, ownership: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        income_in_millions: np.ndarray = income / 1_000_000

        (alpha, beta, C), _ = scipy.optimize.curve_fit(
            sp.lambdify([self.income, self.alpha, self.beta, self.C], self.ownership),
            income_in_millions,
            ownership,
            p0=[1.5, 1, 0.1],
        )
        return {self.alpha: alpha, self.beta: beta / 1_000_000, self.C: C}
