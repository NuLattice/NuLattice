# NOTE(vivek): not a physics guy, so please confirm these tests make sense in the first place
# I assumed that the current implementation is the correct reference

def test_get_lattice_equivalence():
    """Verify that vectorized mgrid lattice generation matches naive loops."""
    from NuLattice.lattice import _get_lattice_list, _get_lattice_mgrid

    myL = 4
    naive_lat = _get_lattice_list(myL)
    opt_lat = _get_lattice_mgrid(myL)
    
    assert len(naive_lat) == len(opt_lat)
    assert naive_lat == opt_lat

def test_get_sp_basis_equivalence():
    """Verify that vectorized basis generation matches original nested loops."""
    from NuLattice.lattice import _get_sp_basis_mgrid, _get_sp_basis_list

    myL = 3
    naive_basis = _get_sp_basis_list(myL)
    opt_basis = _get_sp_basis_mgrid(myL).tolist()
    
    assert len(naive_basis) == len(opt_basis)
    assert naive_basis == opt_basis

def test_state2index_consistency():
    from NuLattice.lattice import get_sp_basis, state2index 
    """Verify state2index correctly maps specific states to their basis position."""

    myL = 4
    basis = get_sp_basis(myL)
    
    # Check first, middle, and last states
    test_indices = [0, len(basis) // 2, len(basis) - 1]
    for idx in test_indices:
        state = basis[idx]
        assert state2index(state, myL) == idx

def test_state2index_equivalence():
    from NuLattice.lattice import _state2index_original, _state2index_strided, get_sp_basis
    """Verify state2index correctly maps specific states to their basis position."""

    myL = 4
    basis = get_sp_basis(myL)

    for state in basis:
        assert _state2index_original(state, myL) == _state2index_strided(state, myL)
    

def test_boundary_conditions():
    """Confirm right/left logic remains consistent for periodic boundaries."""
    from NuLattice.lattice import right, left

    myL = 5

    assert right(4, myL) == 0
    assert right(2, myL) == 3

    assert left(0, myL) == 4
    assert left(3, myL) == 2

def test_kinetic_energy_matrix():
    """Ensure Tkin correctly populates matrix elements for a small lattice."""
    from NuLattice.lattice import get_lattice, Tkin

    myL = 2
    lattice_sites = get_lattice(myL)
    mat = Tkin(lattice_sites, myL)
    
    # Each diagonal element should be 6.0 (2.0 * 3 dimensions)
    diagonals = [val for i, j, val in mat if i == j]
    assert all(val == 6.0 for val in diagonals)
    
    # For a 2^3 lattice with 2 spin/2 isospin, there are 32 single particle states
    # Matrix elements should be (p, q, value)
    assert all(len(row) == 3 for row in mat)

def test_makeState_normalization():
    """Confirm makeState correctly handles float spin/isospin shifts."""
    from NuLattice.lattice import makeState
    # Protons/Neutrons are often passed as +/- 0.5
    state = makeState(1, 1, 1, 0.5, -0.5)
    assert state == [1, 1, 1, 1, 0]
    
    state_float = makeState(0, 0, 0, -0.5, 0.5)
    assert state_float == [0, 0, 0, 0, 1]

def test_ph_space_partitioning():
    from NuLattice.lattice import get_sp_basis, states2PHSpace
    """Verify that states2PHSpace creates a complete partition of the basis."""
    myL = 2
    basis_size = myL**3 * 4
    # Set the first 4 states as holes
    hole_list = get_sp_basis(myL)[:4]
    
    holes, parts = states2PHSpace(hole_list, myL)
    
    assert len(holes) == 4
    assert len(parts) == basis_size - 4
    # Ensure no overlap
    assert len(set(holes).intersection(set(parts))) == 0
    # Ensure union covers all indices
    assert len(set(holes).union(set(parts))) == basis_size

def test_contacts_ordering():
    from NuLattice.lattice import get_lattice, contacts
    """Check that contact matrix elements respect the i < j and k < l constraints."""
    myL = 2
    vT1, vS1 = 1.0, -1.0
    lattice_sites = get_lattice(myL)
    matele = contacts(vT1, vS1, lattice_sites, myL)
    
    for row in matele:
        i, j, k, l, val = row
        assert i < j
        assert k < l
