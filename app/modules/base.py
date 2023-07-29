import abc
import dataclasses
import itertools
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy as sp
from absl import logging


@dataclasses.dataclass(kw_only=True)  # type: ignore
class BaseModule(abc.ABC):
    @classmethod
    def subs(cls, value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, sp.Basic):
            return value.subs(*args, **kwargs)

        if isinstance(value, list):
            return [cls.subs(_value, *args, **kwargs) for _value in value]

        if isinstance(value, dict):
            return {
                key: cls.subs(_value, *args, **kwargs) for key, _value in value.items()
            }

        raise NotImplementedError

    @abc.abstractmethod
    def output(self) -> Any:
        ...

    def __call__(self, output: Any = None, **inputs: float | sp.Basic) -> Any:
        if output is None:
            output = self.output()

        input_by_symbol: dict[sp.Symbol, sp.Basic] = {
            getattr(self, k): v for k, v in inputs.items()
        }

        return self.subs(output, input_by_symbol)


@dataclasses.dataclass(kw_only=True)  # type: ignore
class Module(BaseModule):
    param_by_symbol: dict[sp.Basic, float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.param_by_symbol is not None

    @abc.abstractmethod
    def _fit(self, *args: Any, **kwargs: Any) -> dict[sp.Basic, float]:
        ...

    def fit(self, bootstrap: bool = True, *args: Any, **kwargs: Any) -> None:
        sampled_index: np.ndarray | None = None

        if bootstrap:
            for arg in itertools.chain(args, kwargs.values()):
                if not isinstance(arg, np.ndarray):
                    continue

                if sampled_index is None:
                    sampled_index = np.random.choice(
                        len(arg), size=len(arg), replace=True
                    )

                if len(arg) != sampled_index.shape[0]:
                    raise ValueError("All arguments must have the same length")

            if sampled_index is None:
                raise ValueError("At least one argument must be provided")

            args = tuple(
                [
                    arg[sampled_index] if isinstance(arg, np.ndarray) else arg
                    for arg in args
                ]
            )
            kwargs = {
                key: arg[sampled_index] if isinstance(arg, np.ndarray) else arg
                for key, arg in kwargs.items()
            }

        self.param_by_symbol = self._fit(*args, **kwargs)

    def __call__(self, output: Any = None, **inputs: float | sp.Basic) -> Any:
        if self.param_by_symbol is None:
            raise RuntimeError("Module has not been fitted")

        if output is None:
            output = self.output()

        input_by_symbol: dict[sp.Symbol, sp.Basic] = {
            getattr(self, k): v for k, v in inputs.items()
        }

        return self.subs(output, input_by_symbol | self.param_by_symbol)


@dataclasses.dataclass(kw_only=True)
class BootstrapModule(Module):
    module: Module
    runs: int = 100
    param_by_symbol_list: list[dict[sp.Basic, float]] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.param_by_symbol_list is not None

    def output(self) -> Any:
        return self.module.output()

    def __call__(  # type: ignore
        self,
        output: Any = None,
        quantile: float | Iterable[float] | None = None,
        run_one: bool = False,
        **inputs: float | sp.Basic,
    ) -> Any:
        if self.param_by_symbol_list is None:
            raise RuntimeError("Module has not been fitted")

        if output is None:
            output = self.output()

        if run_one:
            if quantile is not None:
                logging.warning(
                    "Argument `quantile` is ignored when `run_one` is set to `True`. "
                )

            self.module.param_by_symbol = random.choice(self.param_by_symbol_list)
            return self.module.__call__(output, **inputs)

        _outputs: list[Any] = []
        for param_by_symbol in self.param_by_symbol_list:
            self.module.param_by_symbol = param_by_symbol

            _output = self.module.__call__(output, **inputs)
            _outputs.append(_output)

        if quantile is None:
            return _outputs

        if isinstance(quantile, float):
            quantile = [quantile]

        raise NotImplementedError  # TODO

    def _fit(self, *args: Any, **kwargs: Any) -> list[dict[sp.Basic, float]]:  # type: ignore
        param_by_symbol_list = []
        for run in range(self.runs):
            logging.debug(f"Bootstrap iteration {run + 1}")

            module: Module = self.module
            module.fit(*args, **kwargs)
            assert module.param_by_symbol is not None

            param_by_symbol_list.append(module.param_by_symbol)

        return param_by_symbol_list

    def fit(self, *args: Any, **kwargs: Any) -> None:
        self.param_by_symbol_list = self._fit(*args, **kwargs)
