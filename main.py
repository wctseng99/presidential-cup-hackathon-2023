import pprint

from app.modules import PrivateVehicleModule


def main():
    module = PrivateVehicleModule()

    params: dict = {
        module.d: 14_332,  # km/year
        module.f_e: 0.15,  # kWh/km
        module.f_f: 0.08,  # L/km
        module.ρ_e: 0.081,  # $/kWh
        module.ρ_f: 0.997,  # $/L
        module.M_e: 743,  # $/year
        module.M_f: 694,  # $/year
        module.e: 0.14,  # kg/km
        module.Q: 2_000,  # kg
        module.T: 10,  # year
        module.F_e: 6,  # h
        module.F_f: 0.0833,  # h
        module.I_e: 10,
        module.C: 25_000,  # $
        module.k: 1.16,
        module.i_e: 0,
        module.i_f: 230.75,  # $/year
        module.ε: 0.1,
        module.θ: 0.69,
        module.β_1: 1.211e-5,
        module.β_2: 0.05555,
        module.β_3: 0.01831,
        module.λ_1: 0.5,
        module.λ_2: 0.5,
        module.ΔN_v: 100_000,
    }

    # find the leader solution to the stackelberg game
    ρ_c_val = module.solve(params)
    params.update({module.ρ_c: ρ_c_val})

    # unroll the follower solutions
    results = module.subs(module.to_dict(), params)
    pprint.pprint(results)


if __name__ == "__main__":
    main()
