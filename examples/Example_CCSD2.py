import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(os.path.abspath("")) / ".."))
import NuLattice.lattice as lat
import NuLattice.references as ref
import NuLattice.CCM.coupled_cluster as ccm

# Initialize lattice
thisL = 4   #L*L*L lattice
a_lat = 2.0 # lattice spacing in fm
phys_unit = lat.phys_unit(a_lat)

my_basis = lat.get_sp_basis(thisL)
nstat =  len(my_basis)
print("number of single-particle states =", nstat)
lattice = lat.get_lattice(thisL)
nsite = len(lattice)
print("number of lattice sites =", nsite)

# Compute operators for kinetic energy, two-body contacts, and three-body contact

vT1=-8.0  #S-wave isospin-triplet contact
vS1=-8.0  #S-wave spin-triplet contact
cE = 5.5  #three-body contact


myTkin=lat.Tkin(lattice, thisL)
print("number of matrix elements from kinetic energy", len(myTkin))

mycontact=lat.contacts(vT1, vS1, lattice, thisL)
print("number of matrix elements from two-body contacts", len(mycontact))

my3body=lat.NNNcontact(cE, lattice, thisL)
print("number of matrix elements from three-body contacts", len(my3body))

# reference state
ref_state = ref.ref_16O_gs

#Choose whether or not to have v_pppp and v_ppph sparse or not
sparse = True
# make normal-ordered Hamiltonian
refEn, fock_mats, two_body_int = ccm.get_norm_ordered_ham(
    thisL, ref_state, myTkin, mycontact, my3body, sparse=sparse, NO2B=True)


print("energy of reference:", refEn*phys_unit)
#adding a delta to shift the first iteration, avoiding divide by 0 errors
delta = 0

#Whether or not to print out each iteration of the coupled cluster calculation
verbose = True

#solving the coupled cluster equations until we get to a relative error of 1e-8
corrEn, t1, t2 = ccm.ccsd_solver(fock_mats, two_body_int, eps = 1.e-8, maxSteps = 100, 
                                 max_diis = 10, delta = delta, mixing = 0.5,
                                 sparse=sparse, verbose=verbose, ccs=False)

#converting the energy from lattice units to MeV
gsEn = (corrEn + refEn) * phys_unit

print(f'The ground state energy on a L={thisL} size lattice is {gsEn} MeV')

