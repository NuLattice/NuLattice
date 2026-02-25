import numpy as np
import pytest
import NuLattice.lattice_soa as lsoa
import NuLattice.IMSRG.soa.normal_ordering as nosoa
import NuLattice.references as ref

def original_create_occupations(basis, ref_tuples):
    """The original search-based logic."""
    occs = np.zeros(len(basis))
    for x in ref_tuples:
        i = basis.index(x)
        occs[i] = 1.0
    return occs

@pytest.fixture
def imsrg_setup():
    """Fixture to set up a small lattice basis and a mock reference state."""
    L = 2
    basis = lsoa.get_sp_basis(L)
    # Pick 3 arbitrary states to be the 'occupied' reference
    ref_tuples = ref.ref_3H_gs
    # Map those tuples to their integer indices in the basis
    ref_indices = np.array([basis.index(t) for t in ref_tuples])
    
    return {
        "basis": basis,
        "n_states": len(basis),
        "ref_tuples": ref_tuples,
        "ref_indices": ref_indices
    }

def test_soa_occupations_match_original(imsrg_setup):
    """Verify that the SoA (Numba) logic produces the same mask as the Original logic."""
    expected = original_create_occupations(
        imsrg_setup["basis"], 
        imsrg_setup["ref_tuples"]
    )
    
    actual = nosoa.create_occupations(
        imsrg_setup["n_states"], 
        imsrg_setup["ref_indices"]
    )
    
    np.testing.assert_array_equal(actual, expected)

# def test_occupations_match_original(imsrg_setup):
#     """Verify that the Torch scatter logic matches the original result."""
#     expected = original_create_occupations(
#         imsrg_setup["basis"], 
#         imsrg_setup["ref_tuples"]
#     )
    
#     n_states = imsrg_setup["n_states"]
#     ref_idx_tensor = np.array(imsrg_setup["ref_indices"])
    
#     occs_np = np.zeros(n_states, dtype=np.float64)
#     occs_np[ref_idx_tensor] = 1.0
    
#     np.testing.assert_array_equal(occs_np, expected)

def test_occupations_sum_matches_ref_size(imsrg_setup):
    """A physical check: the sum of occupations must equal the number of particles."""
    n_particles = len(imsrg_setup["ref_tuples"])
    occs = nosoa.create_occupations(
        imsrg_setup["n_states"], 
        imsrg_setup["ref_indices"]
    )
    
    assert np.sum(occs) == pytest.approx(float(n_particles))
