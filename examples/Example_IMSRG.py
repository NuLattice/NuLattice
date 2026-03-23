"""
Script to solve the IMSRG(2) equations with dynamic arguments.
"""

import argparse
import sys
import matplotlib.pyplot as plt

import NuLattice.lattice as lat
from NuLattice.constants import ReferenceState
from NuLattice.IMSRG import normal_ordering
from NuLattice.IMSRG import ode_solver

def main():
    parser = argparse.ArgumentParser(description="Run a NuLattice IMSRG(2) calculation.")

    parser.add_argument("--L", type=int, default=2, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.5, help="Lattice spacing in fm")
    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact (D)")

    parser.add_argument("--s_max", type=float, default=40.0, help="Maximum flow parameter s")
    parser.add_argument("--eta_crit", type=float, default=1e-3, help="Convergence criterion for eta")
    parser.add_argument("--plot", action="store_true", help="Display the energy flow plot")
    
    parser.add_argument("--element", type=str, default="HE3", 
                        help="Reference state key (e.g., HE3, HE4, C12)")

    args = parser.parse_args()

    # Setup Lattice Environment
    phys_unit = lat.phys_unit(args.a_lat)
    basis = lat.get_sp_basis(args.L)
    lattice = lat.get_lattice(args.L)

    print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")

    # Kinetic energy and potential matrix elements
    kin = lat.Tkin(lattice, args.L)
    contact_nn = lat.contacts(args.vT1, args.vS1, lattice, args.L)
    contact_3n = lat.NNNcontact(args.cE, lattice, args.L)

    try:
        ref_state = getattr(ReferenceState, f"{args.element.upper()}_GS")
    except AttributeError:
        print(f"Error: Reference state for '{args.element}' not found in constants.")
        sys.exit(1)

    occs = normal_ordering.create_occupations(basis, ref_state)
    e0, f, gamma = normal_ordering.compute_normal_ordered_hamiltonian_no2b(
        occs, kin, contact_nn, contact_3n
    )

    print(f"Initial E0: {e0 * phys_unit:.4f} MeV")

    e_imsrg, integration_data = ode_solver.solve_imsrg2(
        occs, e0, f, gamma, 
        s_max=args.s_max, 
        eta_criterion=args.eta_crit
    )

    print("-" * 30)
    print(f"Final IMSRG Energy: {e_imsrg * phys_unit:.6f} MeV")
    print(f"Energy (Lattice Units): {e_imsrg:.6f}")

    if args.plot:
        s_vals = [x[0] for x in integration_data]
        e_vals = [x[1] for x in integration_data]
        plt.figure(figsize=(8, 5))
        plt.plot(s_vals, e_vals, label=f"{args.element} Flow")
        plt.xlabel("Flow Parameter (s)")
        plt.ylabel("Energy (Lattice Units)")
        plt.title(f"IMSRG(2) Energy Flow: {args.element}")
        plt.xlim(0, min(10.0, args.s_max))
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
