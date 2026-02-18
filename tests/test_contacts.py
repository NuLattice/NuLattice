import pytest
from numpy.testing import assert_allclose
from NuLattice import lattice

def get_test_params():
    """Returns standard physical parameters for testing."""
    return {
        "vT1": 1.5,
        "vS1": -2.0,
        "myL": 2,
        "spin": 2,
        "isospin": 2
    }

def test_contacts_equivalence():
    """
    Directly compares the original loop-based logic (naive) 
    against the new NumPy vectorized implementation.
    """
    p = get_test_params()
    lat = lattice.get_lattice(p["myL"])
    
    legacy_res = lattice._contacts_original(p["vT1"], p["vS1"], lat, p["myL"])
    opt_res = lattice._contacts_np(p["vT1"], p["vS1"], lat, p["myL"]).tolist()
    
    legacy_res.sort()
    opt_res.sort()
    
    assert len(legacy_res) == len(opt_res)
    assert_allclose(opt_res, legacy_res, atol=1e-15)

def test_contacts_antisymmetry():
    """
    Checks that all returned matrix elements follow the 
    required indexing convention: i < j and k < l.
    """
    p = get_test_params()
    lat = lattice.get_lattice(p["myL"])
    matele = lattice.contacts(p["vT1"], p["vS1"], lat, p["myL"])
    
    for row in matele:
        idx1, idx2, idx3, idx4, val = row
        assert idx1 < idx2, f"Antisymmetry failure in bra: {idx1} not < {idx2}"
        assert idx3 < idx4, f"Antisymmetry failure in ket: {idx3} not < {idx4}"

def test_contacts_conservation_laws():
    """
    Verifies that Tz and Sz are strictly conserved for every matrix element.
    """
    p = get_test_params()
    lat = lattice.get_lattice(p["myL"])
    matele = lattice.contacts(p["vT1"], p["vS1"], lat, p["myL"])
    
    basis = lattice.get_sp_basis(p["myL"])
    
    # Check Tz conservation: tz is index 3
    # Check Sz conservation: sz is index 4
    for row in matele:
        x, y, z, tz, sz = row
        s_i, s_j, s_k, s_l = basis[int(x)], basis[int(y)], basis[int(z)], basis[int(tz)]
        
        assert s_i[3] + s_j[3] == s_k[3] + s_l[3], "Tz conservation violated"
        assert s_i[4] + s_j[4] == s_k[4] + s_l[4], "Sz conservation violated"

def test_contacts_onsite_restriction():
    """
    Ensures that contact interactions only occur between particles 
    at the exact same spatial coordinate (i, j, k).
    """
    p = get_test_params()
    lat = lattice.get_lattice(p["myL"])
    matele = lattice.contacts(p["vT1"], p["vS1"], lat, p["myL"])
    basis = lattice.get_sp_basis(p["myL"])
    
    for row in matele:
        x, y, z, tz, _ = row
        coord_i = basis[int(x)][:3]
        coord_j = basis[int(y)][:3]
        coord_k = basis[int(z)][:3]
        coord_l = basis[int(tz)][:3]
        
        assert coord_i == coord_j == coord_k == coord_l, "Interaction is not onsite!"

