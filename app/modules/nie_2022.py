# Nie, Qingyun, Lihui Zhang, and Songrui Li.
#   "How can personal carbon trading be applied in electric vehicle subsidies?
#   A Stackelberg game method in private vehicles." Applied Energy 313 (2022): 118855.
#
# https://www.sciencedirect.com/science/article/abs/pii/S0306261922002914


import enum
from typing import Any

import sympy as sp
import sympy.stats as sps

from app.modules.base import BaseModule


class VehicleSubsidyModule(BaseModule):
    class Density(str, enum.Enum):
        UNIFORM = "UNIFORM"
        NORMAL = "NORMAL"
        
    # `__init__` is a special method used to initialize new instances of the VehicleSubsidyModule class.
    # If no value for `density` is provided when creating a new object, its default value will be `Density.NORMAL`.
    
    def __init__(self, density: Density = Density.NORMAL) -> None:
        d = sp.Symbol("d")
        f_e = sp.Symbol("f_e")
        f_f = sp.Symbol("f_f")
        ρ_e = sp.Symbol("ρ_e")
        ρ_f = sp.Symbol("ρ_f")
        M_e = sp.Symbol("M_e")
        M_f = sp.Symbol("M_f")
        e = sp.Symbol("e")
        Q = sp.Symbol("Q")
        T = sp.Symbol("T")
        F_e = sp.Symbol("F_e")
        F_f = sp.Symbol("F_f")
        I_e = sp.Symbol("I_e")
        C = sp.Symbol("C")
        k = sp.Symbol("k")
        i_e = sp.Symbol("i_e")
        i_f = sp.Symbol("i_f")
        ε = sp.Symbol("ε")
        θ = sp.Symbol("θ")
        β_1 = sp.Symbol("β_1")
        β_2 = sp.Symbol("β_2")
        β_3 = sp.Symbol("β_3")
        λ_1 = sp.Symbol("λ_1")
        λ_2 = sp.Symbol("λ_2")
        N_c = sp.Symbol("N_c")
        ΔN_v = sp.Symbol("ΔN_v")

        ρ_c = sp.Symbol("ρ_c")

        # Equation 1
        S_e = Q * ρ_c
        S_f = (Q - d * e) * ρ_c

        # Equation 10
        l = β_2 * F_f

        # Equation 11
        m = d * f_e * ρ_e - d * f_f * ρ_f + M_e - M_f

        # Equation 12
        n = β_2 * (F_e - F_f) - β_3 * I_e

        # Equation 13
        r = d * f_f * ρ_f + M_f

        # Equation 18
        P1_e = (1 - l - n + ε + β_1 * (k * C + T * (S_e - m - r))) / (2 * β_1)
        P2_e = T * (S_e - m - r) + (ε + 1 - n - l) / β_1
        P1_f = (1 - l + β_1 * (C + T * (S_f - r))) / (2 * β_1)
        P2_f = T * (S_f - r) + (1 - l) / β_1

        # Equation 17
        # This is the solution to the stackelberg game, which in the paper is using `Pm_e` and
        # `Pm_f` to denote, but this is the only operating point for `P_e` and `P_f` we will use
        P_e = sp.Piecewise(
            (P1_e, P2_e - k * C > 0),
            (P2_e, True), #P2_e - k * C < 0
        )
        P_f = sp.Piecewise(
            (P1_f, P2_f - C > 0),
            (P2_f, True), #P2_f - C < 0
        )

        # Equation 8
        θ_1 = β_1 * (P_f + T * (r - S_f)) + l

        # Equation 9
        θ_2 = (β_1 * (P_e - P_f + T * (S_f - S_e + m)) + n) / ε

        # Section 3.2. Solution of stackelberg equilibrium
        # Equation 14
        # Using the `match` statement to handle different values of the `density` variable
        match density:
            # Branch for the scenario where the density parameter is set to 'UNIFORM'
            case VehicleSubsidyModule.Density.UNIFORM:
                η_e = 1 - θ_2
                η_f = θ_2 - θ_1
            # Branch for the scenario where the density parameter is set to 'NORMAL'
            case VehicleSubsidyModule.Density.NORMAL:
                # Creating a Cumulative Distribution Function (CDF) using a LogNormal distribution,
                # where the logarithmically transformed values have a mean of -1.0 and a standard deviation of 0.4
                cdf_fn = sps.cdf(sps.LogNormal("θ", -1.0, 0.4))

                η_e = 1 - cdf_fn(θ_2)
                η_f = sp.Piecewise(
                    (cdf_fn(θ_2) - cdf_fn(θ_1), θ_2 - θ_1 > 0),
                    (0, True),
                )

        # Equation 7
        q_e = η_e * N_c
        q_f = η_f * N_c

        # Equation 2
        V_E = (
            (1 + ε) * θ
            - β_1 * (P_e + (d * f_e * ρ_e + M_e - S_e) * T)
            - β_2 * F_e
            + β_3 * I_e
        )

        # Equation 3
        V_F = θ - β_1 * (P_f + (d * f_f * ρ_f + M_f - S_f) * T) - β_2 * F_f

        # Equation 4
        U_S = (P_e - k * C) * q_e + (P_f - C) * q_f

        # Equation 5
        U_G = -λ_1 * (S_e * q_e + S_f * q_f) - λ_2 * (i_e * q_e + i_f * q_f)

        # Equation 23
        χ_e = η_e / (η_e + η_f)
        χ_f = η_f / (η_e + η_f)

        # Equation 22
        E_G = (S_e + i_e) * χ_e * ΔN_v + (S_f + i_f) * χ_f * ΔN_v

        # Equation 24
        TS = S_e * χ_e * ΔN_v + S_f * χ_f * ΔN_v
        φ = TS / (χ_e * ΔN_v)

        # Equation 25
        CER = χ_e * ΔN_v * d * e
        δ = CER / TS

        # `self` represents the instance of the class and is used to access its attributes and methods.
        # For example, `self.d = d` stores the value of the local variable `d` into the instance's `d` attribute.
        # This ensures that the value of `d` can be accessed and manipulated throughout the instance's lifetime using `self.d`.

        self.d = d
        self.f_e = f_e
        self.f_f = f_f
        self.ρ_e = ρ_e
        self.ρ_f = ρ_f
        self.M_e = M_e
        self.M_f = M_f
        self.e = e
        self.Q = Q
        self.T = T
        self.F_e = F_e
        self.F_f = F_f
        self.I_e = I_e
        self.C = C
        self.k = k
        self.i_e = i_e
        self.i_f = i_f
        self.ε = ε
        self.θ = θ
        self.β_1 = β_1
        self.β_2 = β_2
        self.β_3 = β_3
        self.λ_1 = λ_1
        self.λ_2 = λ_2
        self.N_c = N_c
        self.ΔN_v = ΔN_v

        self.ρ_c = ρ_c
        self.S_e = S_e
        self.S_f = S_f
        self.l = l
        self.m = m
        self.n = n
        self.r = r
        self.P1_e = P1_e
        self.P2_e = P2_e
        self.P1_f = P1_f
        self.P2_f = P2_f
        self.P_e = P_e
        self.P_f = P_f
        self.θ_1 = θ_1
        self.θ_2 = θ_2
        self.η_e = η_e
        self.η_f = η_f
        self.q_e = q_e
        self.q_f = q_f
        self.V_E = V_E
        self.V_F = V_F
        self.U_S = U_S
        self.U_G = U_G
        self.χ_e = χ_e
        self.χ_f = χ_f
        self.E_G = E_G
        self.TS = TS
        self.φ = φ
        self.CER = CER
        self.δ = δ


    # 'output' is designed to return a dictionary, the values are of type 'sp.Basic'

    def output(self) -> dict[str, sp.Basic]:
        return {
            "ρ_c": self.ρ_c,
            "subsidy_unit_price": self.ρ_c,
            "S_e": self.S_e,
            "electric_vehicle_subsidy": self.S_e,
            "S_f": self.S_f,
            "fuel_vehicle_subsidy": self.S_f,
            "l": self.l,
            "m": self.m,
            "n": self.n,
            "r": self.r,
            "P1_e": self.P1_e,
            "P2_e": self.P2_e,
            "P1_f": self.P1_f,
            "P2_f": self.P2_f,
            "P_e": self.P_e,
            "electric_vehicle_purchasing_price": self.P_e,
            "P_f": self.P_f,
            "fuel_vehicle_purchasing_price": self.P_f,
            "θ_1": self.θ_1,
            "θ_2": self.θ_2,
            "η_e": self.η_e,
            "electric_vehicle_demand_rate": self.η_e,
            "η_f": self.η_f,
            "fuel_vehicle_demand_rate": self.η_f,
            "q_e": self.q_e,
            "electric_vehicle_demand": self.q_e,
            "q_f": self.q_f,
            "fuel_vehicle_demand": self.q_f,
            "V_E": self.V_E,
            "electric_vehicle_utility": self.V_E,
            "V_F": self.V_F,
            "fuel_vehicle_utility": self.V_F,
            "U_S": self.U_S,
            "supplier_utility": self.U_S,
            "U_G": self.U_G,
            "government_utility": self.U_G,
            "χ_e": self.χ_e,
            "electric_vehicle_normalized_demand_rate": self.χ_e,
            "χ_f": self.χ_f,
            "fuel_vehicle_normalized_demand_rate": self.χ_f,
            "E_G": self.E_G,
            "total_government_expenditure": self.E_G,
            "TS": self.TS,
            "total_government_subsidy_expenditure": self.TS,
            "φ": self.φ,
            "subsidy_expenditure_rate": self.φ,
            "CER": self.CER,
            "total_emmision_recuction": self.CER,
            "δ": self.δ,
            "emission_reduction_unit_cost": self.δ,
        }

    def __call__(self, output: Any = None, **inputs: sp.Basic) -> Any:
        # Extract ρ_c from the inputs; if not present, default to None.
        ρ_c = inputs.get("ρ_c", None)

        # If ρ_c is not provided in the inputs, we'll need to compute it.
        if ρ_c is None:
            
            # This suggests that the function represents a leader-follower game, often termed as Stackelberg game. 
            # The leader (in this case, the government) makes the first move, and the follower reacts accordingly.
            # Calculate the derivative of U_G with respect to ρ_c. This derivative represents 
            # how the utility of the government (U_G) changes with respect to ρ_c.

            # Section 3.2. Solution of stackelberg equilibrium, yield the optimal government subsidy price
            dU_G__dρ_c = super().__call__(output=sp.diff(self.U_G, self.ρ_c), **inputs)
            
            # Solve for ρ_c such that the derivative is zero. These points (ρ_c values) represent potential 
            # maxima or minima for the government's utility.
            
            ρ_c: list[sp.Basic] = sp.solve(dU_G__dρ_c, self.ρ_c)

            # Filter out any negative solutions for ρ_c, as they might be non-feasible in this context.
            ρ_c = [ρ_c_ for ρ_c_ in ρ_c if ρ_c_ > 0]

            # Check if there's exactly one feasible solution for ρ_c. 
            # If not, raise an error because the model expects a unique solution.
            if len(ρ_c) != 1:
                raise ValueError(
                    f"Expected one solution for ρ_c, found {len(ρ_c)}: {ρ_c}"
                )
            # Update the inputs with the calculated value of ρ_c.
            inputs.update({"ρ_c": ρ_c[0]})

        # This might involve evaluating some other functions or equations that are defined in the parent class.
        # unroll the follower solutions
        return super().__call__(output=output, **inputs)
