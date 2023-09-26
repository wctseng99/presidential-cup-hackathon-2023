from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize
import sklearn.metrics
import sympy as sp
from absl import logging

from app.modules.base import Module

# The following classes define different mathematical modules used for modeling and fitting specific curve types.


# LinearModule Models a linear relationship between input variables.
class LinearModule(Module):
    def __init__(self, input_dims: int = 1, bias: bool = True):
        x = sp.symarray("x", input_dims)
        a = sp.symarray("a", input_dims)
        b = sp.Symbol("b")

        if bias:
            y = np.dot(x, a) + b
        else:
            y = np.dot(x, a)

        self.input_dims = input_dims
        self.bias = bias

        self.x = x
        self.a = a
        for d in range(input_dims):
            setattr(self, f"x_{d}", x[d])
            setattr(self, f"a_{d}", a[d])
        self.b = b

        self.y = y

    def output(self) -> dict[str, sp.Basic]:
        # Returns the output variables, including coefficients 'a' and bias 'b'.
        return {f"a_{d}": self.a[d] for d in range(self.input_dims)} | {
            "b": self.b,
            "y": self.y,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[sp.Basic, float]:  # type: ignore
        # Fits the LinearModule to input data and returns fitted parameters.
        (num_samples, input_dims) = X.shape
        if input_dims != self.input_dims:
            raise ValueError(f"Expected {self.input_dims} inputs, but got {input_dims}")

        a: np.ndarray
        b: float
        y_pred: np.ndarray
        if self.bias:
            X_one = np.r_["1,2,0", X, np.ones(num_samples)]
            a_one, _, _, _ = scipy.linalg.lstsq(X_one, y)
            a = a_one[:-1]
            b = a_one[-1]
            y_pred = X_one @ a_one
        else:
            a, _, _, _ = scipy.linalg.lstsq(X, y)
            b = 0
            y_pred = X @ a

        r2: float = sklearn.metrics.r2_score(y, y_pred)

        logging.debug(f"Fitted with r2={r2} and parameters: a={a}, b={b}")
        return {self.a[d]: a[d] for d in range(self.input_dims)} | {self.b: b}


# SigmoidCurveModule models a sigmoid curve.
class SigmoidCurveModule(Module):
    def __init__(self):
        x = sp.Symbol("x")
        gamma = sp.Symbol("gamma")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")

        y = gamma * (1 - alpha * (1 - 1 / (1 + sp.exp(-beta * x))))

        self.x = x
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.y = y

    def output(self) -> dict[str, sp.Basic]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "y": self.y,
        }

    def _fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[sp.Basic, float]:  # type: ignore
        fn: Callable[[np.ndarray, float, float, float], np.ndarray] = sp.lambdify(
            [self.x, self.gamma, self.alpha, self.beta], self.y
        )
        (gamma, alpha, beta), _ = scipy.optimize.curve_fit(fn, x, y, **kwargs)
        y_pred: np.ndarray = np.vectorize(fn)(x, gamma, alpha, beta)
        r2: float = sklearn.metrics.r2_score(y, y_pred)

        logging.debug(
            f"Fitted with r2={r2} and parameters: gamma={gamma}, alpha={alpha}, beta={beta}"
        )
        return {self.gamma: gamma, self.alpha: alpha, self.beta: beta}


# GompertzCurveModule models a Gompertz curve.
class GompertzCurveModule(Module):
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

    def output(self) -> dict[str, sp.Basic]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "y": self.y,
        }

    def _fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[sp.Basic, float]:  # type: ignore
        fn: Callable[[np.ndarray, float, float, float], np.ndarray] = sp.lambdify(
            [self.x, self.gamma, self.alpha, self.beta], self.y
        )
        (gamma, alpha, beta), _ = scipy.optimize.curve_fit(fn, x, y, **kwargs)
        y_pred: np.ndarray = np.vectorize(fn)(x, gamma, alpha, beta)
        r2: float = sklearn.metrics.r2_score(y, y_pred)

        logging.debug(
            f"Fitted with r2={r2} and parameters: gamma={gamma}, alpha={alpha}, beta={beta}"
        )
        return {self.gamma: gamma, self.alpha: alpha, self.beta: beta}


# GammaCurveModule models a gamma distribution curve.
class GammaCurveModule(Module):
    def __init__(self):
        x = sp.Symbol("x")
        alpha = sp.Symbol("alpha")
        beta = sp.Symbol("beta")
        C = sp.Symbol("C")

        y = (
            (beta**alpha) * (x ** (alpha - 1)) * sp.exp(-beta * x) / sp.gamma(alpha)
        ) + C

        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.C = C

        self.y = y

    def output(self) -> dict[str, sp.Basic]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "C": self.C,
            "y": self.y,
        }

    def _fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[sp.Basic, float]:  # type: ignore
        fn: Callable[[np.ndarray, float, float, float], np.ndarray] = sp.lambdify(
            [self.x, self.alpha, self.beta, self.C], self.y
        )
        (alpha, beta, C), _ = scipy.optimize.curve_fit(fn, x, y, **kwargs)
        y_pred: np.ndarray = np.vectorize(fn)(x, alpha, beta, C)
        r2: float = sklearn.metrics.r2_score(y, y_pred)

        logging.debug(
            f"Fitted with r2={r2} and parameters: alpha={alpha}, beta={beta}, C={C}"
        )
        return {self.alpha: alpha, self.beta: beta, self.C: C}
