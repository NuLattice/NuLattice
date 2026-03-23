import argparse
import sys

import numpy as np

import NuLattice.HF.hartree_fock as hf
import NuLattice.lattice as lat
from NuLattice.constants import ReferenceState

def main():
    parser = argparse.ArgumentParser(description="Run a NuLattice Hartree-Fock calculation.")

    parser.add_argument("--L", type=int, default=8, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.5, help="Lattice spacing in fm")
    
    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1e-8, help="Convergence threshold")
    parser.add_argument("--mix", type=float, default=0.7, help="Mixing parameter for density iterations")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum HF iterations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")

    parser.add_argument("--element", type=str, default="O16", 
                        help="Reference state key (e.g., O16, C12, HE4)")

    args = parser.parse_args()

    phys_unit = lat.phys_unit(args.a_lat)
    my_basis = lat.get_sp_basis(args.L)
    lattice = lat.get_lattice(args.L)
    
    nstat = len(my_basis)
    nsite = len(lattice)

    print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")
    print(f"SP States: {nstat} | Lattice Sites: {nsite}")

    myTkin = lat.Tkin(lattice, args.L)
    mycontact = lat.contacts(args.vT1, args.vS1, lattice, args.L)
    my3body = lat.NNNcontact(args.cE, lattice, args.L)

    print(f"Matrix elements - Tkin: {len(myTkin)}, 2-body: {len(mycontact)}, 3-body: {len(my3body)}")

    try:
        attr_name = f"{args.element.upper()}_GS"
        my_ref = getattr(ReferenceState, attr_name)
    except AttributeError:
        print(f"Error: Reference state for '{args.element}' not found.")
        sys.exit(1)

    hole = ReferenceState.holes(my_ref, my_basis)
    hnum = len(hole)

    dens = hf.init_density(nstat, hole)
    print(f"Target Particle Number: {hnum} | Initial Trace: {np.trace(dens)}")

    erg, trafo, conv = hf.solve_HF(
        myTkin,
        mycontact,
        my3body,
        dens,
        mix=args.mix,
        eps=args.eps,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )

    print("-" * 30)
    if conv:
        final_energy = erg * phys_unit
        print("HF Convergence: SUCCESS")
        print(f"Final HF Energy: {final_energy:.6f} MeV")
    else:
        print("HF Convergence: FAILED (Check mixing or max_iter)")

if __name__ == "__main__":
    main()
