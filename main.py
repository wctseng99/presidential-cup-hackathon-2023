import pprint

from app.modules import VehicleSubsidyModule


def main():
    module = VehicleSubsidyModule()

    params: dict = {
        "d": 14_332,  # km/year
        "f_e": 0.15,  # kWh/km
        "f_f": 0.08,  # L/km
        "ρ_e": 0.081,  # $/kWh
        "ρ_f": 0.997,  # $/L
        "M_e": 743,  # $/year
        "M_f": 694,  # $/year
        "e": 0.14,  # kg/km
        "Q": 2_000,  # kg
        "T": 10,  # year
        "F_e": 6,  # h
        "F_f": 0.0833,  # h
        "I_e": 10,
        "C": 25_000,  # $
        "k": 1.16,
        "i_e": 0,
        "i_f": 230.75,  # $/year
        "ε": 0.1,
        "θ": 0.69,
        "β_1": 1.211e-5,
        "β_2": 0.05555,
        "β_3": 0.01831,
        "λ_1": 0.5,
        "λ_2": 0.5,
        "ΔN_v": 100_000,
    }

    results = module.forward(params)
    pprint.pprint(results)


if __name__ == "__main__":
    main()
