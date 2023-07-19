import abc
import itertools
from typing import Any

import numpy as np
import sympy as sp


class Module(abc.ABC):
    def subs(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, sp.Basic):
            return value.subs(*args, **kwargs)

        if isinstance(value, list):
            return [self.subs(_value, *args, **kwargs) for _value in value]

        if isinstance(value, dict):
            return {
                key: self.subs(_value, *args, **kwargs) for key, _value in value.items()
            }

        raise NotImplementedError

    @abc.abstractmethod
    def output_symbols(self) -> dict[str, sp.Basic]:
        ...

    def _forward(
        self,
        output_symbols: dict[str, sp.Basic],
        input_values: dict[str, float | sp.Basic],
    ) -> dict[str, sp.Basic]:
        input_symbol_values: dict[sp.Basic, float | sp.Basic] = {
            getattr(self, k): v for k, v in input_values.items()
        }
        return self.subs(output_symbols, input_symbol_values)

    def forward(self, input_values: dict[str, float | sp.Basic]) -> dict[str, sp.Basic]:
        return self._forward(
            output_symbols=self.output_symbols(), input_values=input_values
        )


class FittableModule(Module):
    @classmethod
    def bootstrap(cls, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        size: int | None = None
        for arg in itertools.chain(args, kwargs.values()):
            if size is None:
                size = len(arg)

            if len(arg) != size:
                raise ValueError("All arguments must have the same length")

        if size is None:
            raise ValueError("At least one argument must be provided")

        index = np.arange(size)
        sampled_index = np.random.choice(index, size=size, replace=True)

        return (
            [arg[sampled_index] for arg in args],
            {key: arg[sampled_index] for key, arg in kwargs.items()},
        )

    def __init__(self):
        self._param_symbol_values: dict[sp.Basic, float] | None = None

    @abc.abstractmethod
    def _fit(self, *args: Any, **kwargs: Any) -> dict[sp.Basic, float]:
        ...

    def fit(self, bootstrap: bool = True, *args: Any, **kwargs: Any) -> None:
        if bootstrap:
            args, kwargs = self.bootstrap(*args, **kwargs)

        self.param_symbol_values = self._fit(*args, **kwargs)

    def _forward(
        self,
        output_symbols: dict[str, sp.Basic],
        input_values: dict[str, float | sp.Basic],
    ) -> dict[str, sp.Basic]:
        if self.param_symbol_values is None:
            raise RuntimeError("Module has not been fitted")

        input_symbol_values: dict[sp.Basic, float | sp.Basic] = {
            getattr(self, k): v for k, v in input_values.items()
        }
        return self.subs(
            output_symbols, {**input_symbol_values, **self.param_symbol_values}
        )
