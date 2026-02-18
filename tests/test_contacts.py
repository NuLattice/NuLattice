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

def test_nnn_contact_equivalence():
    """Verify optimized NNNcontact matches naive implementation."""
    myL = 2
    v3NF = 0.5
    lat = lattice.get_lattice(myL)
    
    naive_res = lattice._NNNcontact_original(v3NF, lat, myL)
    opt_res = lattice._NNNcontact_np(v3NF, lat, myL).tolist()
    
    naive_res.sort()
    opt_res.sort()
    
    assert len(naive_res) == len(opt_res)
    assert_allclose(opt_res, naive_res, atol=1e-15)

def test_nnn_contact_antisymmetry():
    """Check i1 < i2 < i3 constraint."""
    myL = 2
    lat = lattice.get_lattice(myL)
    matele = lattice.NNNcontact(1.0, lat, myL)
    
    for row in matele:
        # Indices: i1, i2, i3, j1, j2, j3, val
        assert row[0] < row[1] < row[2]
        assert row[3] < row[4] < row[5]
        # Diagonal check
        assert row[0] == row[3] and row[1] == row[4] and row[2] == row[5]

def test_tkin_equivalence():
    """Compare naive loop-based Tkin with vectorized implementation."""
    myL = 3
    lat_sites = lattice.get_lattice(myL)
    
    legacy_res = lattice._Tkin_original(lat_sites, myL)
    opt_res = lattice._Tkin_np(lat_sites, myL).tolist()
    res_boogaloo = lattice._Tkin_np_flat(lat_sites, myL).tolist()
    
    legacy_res.sort()
    opt_res.sort()
    res_boogaloo.sort()
    
    assert len(legacy_res) == len(opt_res)
    assert len(legacy_res) == len(res_boogaloo)
    assert_allclose(opt_res, legacy_res, atol=1e-15)
    assert_allclose(res_boogaloo, legacy_res, atol=1e-15)

def test_tkin_diagonals():
    """In 3D, each diagonal element must be 2.0 * 3 = 6.0."""
    myL = 4
    lat = lattice.get_lattice(myL)
    mat = lattice.Tkin(lat, myL)
    
    diagonals = [row[2] for row in mat if row[0] == row[1]]
    # Total single particle states = L^3 * spin * isospin
    expected_count = (myL**3) * 4 
    
    assert len(diagonals) == expected_count
    assert all(val == 6.0 for val in diagonals)

def test_tkin_hermiticity():
    """T_ij must equal T_ji (since matrix elements are real)."""
    myL = 3
    lat = lattice.get_lattice(myL)
    mat = lattice.Tkin(lat, myL)
    
    # Convert list of [p, q, val] to a dictionary for fast lookup
    lookup = {(int(p), int(q)): val for p, q, val in mat}
    
    for (p, q), val in lookup.items():
        assert (q, p) in lookup, f"Missing Hermitian partner for ({p}, {q})"
        assert_allclose(lookup[(q, p)], val, atol=1e-15)

def test_tkin_sparsity_count():
    """
    On a 3D lattice, each state has 1 diagonal + 6 neighbors (2 per dimension).
    Total elements should be N_states * 7.
    """
    myL = 3
    lat = lattice.get_lattice(myL)
    n_states = (myL**3) * 4
    mat = lattice.Tkin(lat, myL)
    
    assert len(mat) == n_states * 7

def test_tkin_offdiagonal_values():
    """All hopping elements must be -1.0."""
    myL = 2
    lat = lattice.get_lattice(myL)
    mat = lattice.Tkin(lat, myL)
    
    off_diagonals = [row[2] for row in mat if row[0] != row[1]]
    assert all(val == -1.0 for val in off_diagonals)
