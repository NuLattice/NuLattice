import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.lattice as lat
import NuLattice.references as ref
import NuLattice.CCM.coupled_cluster as ccm

#Defining the lattice, in our case a 2x2x2 lattice with lattice spacing of 2.5 fm
thisL = 2
a_lat  = 2.5

#two-body contacts
vT1 = -9.0
vS1 = -9.0
#three-body contact
w3  = 6.0 

# reference state
ref_state = ref.ref_16O_gs

#Choose whether or not to have v_pppp and v_ppph sparse or not
sparse = True
# make normal-ordered Hamiltonian
refEn, fock_mats, two_body_int = ccm.get_norm_ord_int(
    thisL, ref_state, vT1, vS1, w3, sparse=sparse)

#Whether or not to print out each iteration of the coupled cluster calculation
verbose = True

#solving the coupled cluster equations until we get to a relative error of 1e-8
corrEn, t1, t2 = ccm.ccsd_solver(fock_mats, two_body_int, 
                            eps = 1e-8, maxSteps = 500, max_diis = 10, mixing = 0.5,
                            sparse=sparse, verbose=verbose)

#converting the energy from lattice units to MeV
phys_unit = lat.phys_unit(a_lat)
gsEn = (corrEn + refEn) * phys_unit

print(f'The ground state energy of the O16 on a L={thisL} size lattice is {gsEn} MeV')
