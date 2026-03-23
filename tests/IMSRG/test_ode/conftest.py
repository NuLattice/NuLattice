import pytest
import numpy as np
import torch
import NuLattice.lattice_soa as lsoa
import NuLattice.references as ref
from NuLattice.IMSRG.soa import normal_ordering as nosoa

@pytest.fixture(scope="module")
def helium3_hamiltonian():
    """Sets up the He3 Hamiltonian on a L=2 lattice."""
    L, a_lat = 2, 2.5
    phys_unit = lsoa.phys_unit(a_lat)
    basis = lsoa.get_sp_basis(L)
    lattice = lsoa.get_lattice(L)
    
    # Couplings
    vT1, vS1, D = -9.0, -9.0, 6.0
    
    # Operators
    kin = lsoa.Tkin(lattice, L)
    contact_nn = lsoa.contacts(vT1, vS1, lattice, L)
    contact_3n = lsoa.NNNcontact(D, lattice, L)
    
    # Occupations (SoA way)
    he3_ref = np.array(ref.ref_3He_gs)
    occs = nosoa.create_occupations(np.array(basis), he3_ref)
    
    # Normal Order
    e0, f, gamma = nosoa.compute_normal_ordered_hamiltonian_no2b(
        occs, kin, contact_nn, contact_3n
    )
    
    return {
        "occs": torch.tensor(occs, dtype=torch.float32),
        "e0": e0,
        "f": torch.tensor(f, dtype=torch.float32),
        "gamma": torch.tensor(gamma, dtype=torch.float32),
        "phys_unit": phys_unit
    }
