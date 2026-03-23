import numpy as np
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent / ".."))


import NuLattice.HF.soa.hartree_fock as hf
import NuLattice.lattice_soa as lat
import NuLattice.references as ref
import torch

device = torch.device('cuda')
# Initialize lattice
thisL = 4  # L*L*L lattice
a_lat = 2.5  # lattice spacing in fm
phys_unit = lat.phys_unit(a_lat)

my_basis = lat.get_sp_basis(thisL)
nstat = len(my_basis)
print("number of single-particle states =", nstat)
lattice = lat.get_lattice(thisL)
nsite = len(lattice)
print("number of lattice sites =", nsite)

# Compute operators for kinetic energy, two-body contacts, and three-body contact

vT1 = -9.0  # S-wave isospin-triplet contact
vS1 = -9.0  # S-wave spin-triplet contact
cE = 6.0  # three-body contact


myTkin = lat._Tkin(lattice, thisL)
print("number of matrix elements from kinetic energy", len(myTkin))

mycontact = lat._contacts(vT1, vS1, lattice, thisL)
print("number of matrix elements from two-body contacts", len(mycontact))

my3body = lat._NNNcontact(cE, lattice, thisL)
print("number of matrix elements from three-body contacts", len(my3body))

# we compute oxygen-16
my_ref = ref.ref_16O_gs
hole = ref.reference_to_holes(my_ref, my_basis)
hnum = len(hole)

dens = hf.init_density(nstat, hole)
print("number of particles:", np.trace(dens), "compare with", hnum)

eps = 1.0e-8
mix = 0.7
max_iter = 100
verbose = True
erg, trafo, conv = hf.solve_HF(
    myTkin.to_torch(device),
    mycontact.to_torch(device),
    my3body.to_torch(device),
    dens.to(device=device),
    mix=mix,
    eps=eps,
    max_iter=max_iter,
    verbose=verbose,
)

if conv:
    print("HF energy (MeV) = ", erg * phys_unit)
else:
    print("HF did not converge")
