import pandas as pd

from app.data.core import Vehicle

# Using pattern matching to determine the appropriate set of k values based on the vehicle type.
# k values means the ratio of production cost of EV to FV
def get_nie_k_series(vehicle: Vehicle) -> pd.Series:
    match vehicle:
        case Vehicle.CAR:
            data = {
                2020: 1.37,
                2021: 1.309,
                2022: 1.261,
                2023: 1.219,
                2024: 1.182,
                2025: 1.147,
                2026: 1.115,
                2027: 1.085,
                2028: 1.057,
                2029: 1.031,
                2030: 1.007,
                2031: 0.984,
                2032: 0.957,
                2033: 0.936,
                2034: 0.916,
                2035: 0.896,
                2036: 0.876,
                2037: 0.857,
                2038: 0.837,
                2039: 0.818,
                2040: 0.799,
                2041: 0.776,
                2042: 0.753,
                2043: 0.734,
                2044: 0.716,
                2045: 0.698,
                2046: 0.68,
                2047: 0.662,
                2048: 0.645,
                2049: 0.627,
                2050: 0.61,
            }
        # For any other vehicle type: Raise an error indicating the given vehicle type is not supported.
        case _:
            raise ValueError(f"Invalid vehicle: {vehicle}")

    s = pd.Series(data=data, name="k")

    return s

# Using pattern matching to determine the target percentage for electric vehicle sales based on the vehicle type.
def get_electric_vehicle_sale_percentage_target_series(vehicle: Vehicle) -> pd.Series:
    match vehicle:
        case Vehicle.CAR:
            data = {
                2023: 0.03,
                2024: 0.04,
                2025: 0.05,
                2026: 0.10,
                2027: 0.15,
                2028: 0.20,
                2029: 0.25,
                2030: 0.30,
                2040: 1.00,
            }
        case _:
            raise ValueError(f"Invalid vehicle: {vehicle}")

    s = pd.Series(data=data, name="electric_vehicle_sale_percentage_target")

    return s
