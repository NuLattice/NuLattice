import argparse
import torch

import NuLattice.lattice_soa as lat
import NuLattice.references as ref
import NuLattice.CCM.soa.coupled_cluster as ccm

device = torch.device('cuda')

def main():
    parser = argparse.ArgumentParser(description="Run a NuLattice CCM calculation with custom parameters.")

    parser.add_argument("--L", type=int, default=4, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.0, help="Lattice spacing in fm")

    parser.add_argument("--vT1", type=float, default=-8.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-8.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=5.5, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1.e-8, help="Convergence threshold")
    parser.add_argument("--maxSteps", type=int, default=100, help="Maximum CC iterations")
    parser.add_argument("--max_diis", type=int, default=10, help="Max DIIS subspace size")
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for iterations")
    parser.add_argument("--delta", type=float, default=0.0, help="Energy shift to avoid division by zero")
    
    parser.add_argument("--sparse", action="store_false", default=True, help="Disable sparse matrices (defaults to True)")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")

    args = parser.parse_args()

    phys_unit = lat.phys_unit(args.a_lat)
    my_basis = lat.get_sp_basis(args.L)
    lattice = lat.get_lattice(args.L)
    
    print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")
    print(f"Number of single-particle states = {len(my_basis)}")
    print(f"Number of lattice sites = {len(lattice)}")

    myTkin = lat.Tkin(lattice, args.L)
    mycontact = lat.contacts(args.vT1, args.vS1, lattice, args.L)
    my3body = lat.NNNcontact(args.cE, lattice, args.L)

    print(f"Matrix elements - Tkin: {len(myTkin)}, 2-body: {len(mycontact)}, 3-body: {len(my3body)}")

    # reference state
    ref_state = ref.ref_16O_gs

    refEn, fock_mats, two_body_int = ccm.get_norm_ordered_ham(
        args.L,
        ref_state,
        myTkin.to_torch(),
        mycontact.to_torch(),
        my3body.to_torch(),
        sparse=args.sparse,
        NO2B=True
    )

    print(f"Energy of reference: {refEn*phys_unit} MeV")

    corrEn, t1, t2 = ccm.ccsd_solver(
        fock_mats, 
        two_body_int, 
        eps=args.eps, 
        maxSteps=args.maxSteps, 
        max_diis=args.max_diis, 
        delta=args.delta, 
        mixing=args.mixing,
        sparse=args.sparse, 
        verbose=args.verbose, 
        ccs=False
    )

    gsEn = (corrEn + refEn) * phys_unit
    print("-" * 30)
    print(f"Final Ground State Energy: {gsEn:.6f} MeV")

if __name__ == "__main__":
    main()
