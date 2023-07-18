from typing import Any

import sympy as sp


class Module(object):
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
