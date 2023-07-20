import numpy as np
import scipy.optimize
import sympy as sp
import sympy.stats.rv

from app.modules.base import FittableModule


class GompertzDistributionModule(FittableModule):
    def __init__(self):
        x = sp.Symbol("x")
        gamma = sp.Symbol("gamma")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")

        y = gamma * sp.exp(alpha * sp.exp(beta * x))

        self.x = x
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.y = y

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "y": self.y,
        }

    def _fit(self, x: np.ndarray, y: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        (gamma, alpha, beta), _ = scipy.optimize.curve_fit(
            sp.lambdify([self.x, self.gamma, self.alpha, self.beta], self.y),
            x,
            y,
            p0=[np.max(y), -2, -2],
        )
        return {self.gamma: gamma, self.alpha: alpha, self.beta: beta}


class GammaDistributionModule(FittableModule):
    def __init__(self):
        x = sp.Symbol("x")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")
        C = sp.Symbol("C")

        y = ((beta**alpha) * (x ** (alpha - 1)) * sp.exp(-beta * x)) / sp.gamma(
            alpha
        ) + C

        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.C = C

        self.y = y

    def output_symbols(self) -> dict[str, sp.Basic]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "C": self.C,
            "y": self.y,
        }

    def _fit(self, x: np.ndarray, y: np.ndarray) -> dict[sp.Basic, float]:  # type: ignore
        (alpha, beta, C), _ = scipy.optimize.curve_fit(
            sp.lambdify([self.x, self.alpha, self.beta, self.C], self.y),
            x,
            y,
            p0=[1.5, 1, 0.1],
        )
        return {self.alpha: alpha, self.beta: beta, self.C: C}
