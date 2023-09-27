# CodeGreen Monorepo for Presidential Cup 2023

This repository contains the code for the Presidential Cup 2023 implementations.

## Prerequisites

- [Poetry](https://python-poetry.org/docs/#installation)

## Installation

```bash
# Clone the repository
$ git clone git@github.com:wctseng99/presidential-cup-hackathon-2023.git && cd presidential-cup-hackathon-2023
# Install dependencies
$ poetry shell && poetry install
```

## Usage

```bash
$ python main.py
```
- The **data** folder is specifically dedicated to the extraction, organization, extrapolation, and output of data required for computational purposes.
- The **module** folder is responsible for constructing simulation models used in experiments.
- The **pipelines** folder serves to connect the data and model pipelines required during simulations.

## References
Nie, Qingyun, Lihui Zhang, and Songrui Li.
[How can personal carbon trading be applied in electric vehicle subsidies? A Stackelberg game method in private vehicles.](https://www.sciencedirect.com/science/article/abs/pii/S0306261922002914) Applied Energy 313 (2022): 118855.

Tsai, Chia-Yu, Tsung-Heng Chang, and I-Yun Lisa Hsieh.
[Evaluating vehicle fleet electrification against net-zero targets in scooter-dominated road transport.](https://www.sciencedirect.com/science/article/pii/S1361920922003686) Transportation Research Part D: Transport and Environment 114 (2023): 103542.

## Libraries

- **dataclasses**: This module provides decorators and functions for automatically adding special methods to user-defined classes. It's often used to create simple classes for storing data.
- **functools**: This module provides higher-order functions and operations on callable objects. It's commonly used for functional programming tasks and can be used to create cached properties with functools.cached_property.
- **pathlib.Path**: This module provides an object-oriented interface for file system paths. 
- **numpy (np)**: NumPy is a library for numerical and array-based operations in Python. It provides support for arrays, matrices, and a wide range of mathematical functions.
- **pandas (pd)**: Pandas is a library for data manipulation and analysis. It provides data structures like DataFrames and Series, which are used for handling and analyzing tabular data.
- **rich**: The rich library is used for adding rich text and progress tracking to command-line interfaces (CLIs).
- **collections.abc**: This module defines abstract base classes (ABCs) for various collections. In this repo, it imports Callable and Iterable, which are abstract base classes for callables (e.g., functions) and iterable objects (e.g., lists), respectively.
- **scipy**: SciPy is a library for scientific and technical computing in Python. The scipy.integrate module provides functions for numerical integration and solving differential equations
- **sympy (sp)**: SymPy is a Python library for symbolic mathematics. It allows you to perform algebraic and symbolic computations, making it useful for symbolic mathematics tasks.
- **absl**: The absl library is used for writing command-line applications in Python and includes functionality for handling command-line arguments and logging. 
- **random**: The random module provides functions to generate random numbers and perform randomization tasks in your Python programs. 