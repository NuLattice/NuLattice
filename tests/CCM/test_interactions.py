import pytest
import numpy as np

import NuLattice.CCM.coupled_cluster as legacy_cc
import NuLattice.CCM.soa.coupled_cluster as soa_cc


@pytest.fixture
def mock_contact_data():
    """Generates randomized contact data for a P=4, H=4 system."""
    # System: 8 total states, 4 holes (indices 0-3), 4 particles (indices 4-7)
    holes = [0, 1, 2, 3]
    parts = [4, 5, 6, 7]

    # Create random contacts [i, j, k, l, val]
    # ensure indices cover p-p, p-h, and h-h combinations
    np.random.seed(42)
    contacts = []
    all_states = holes + parts
    for _ in range(20):
        idx = np.random.choice(all_states, 4, replace=False).tolist()
        val = np.random.uniform(-1, 1)
        contacts.append(idx + [val])

    return parts, holes, contacts


def test_get_all_interactions_dense_parity(mock_contact_data):
    """Verifies numerical parity for blocks that are always dense (pphh, phph, phhh, hhhh)."""
    parts, holes, contacts = mock_contact_data

    leg_res = legacy_cc.get_all_interactions(parts, holes, contacts, sparse=False)

    soa_res = soa_cc.get_all_interactions(parts, holes, contacts, sparse=False)

    block_names = ["v_pphh", "v_phph", "v_phhh", "v_hhhh"]
    for i in range(2, 6):
        leg_block = leg_res[i]
        soa_block = soa_res[i].detach().cpu().numpy()

        assert np.allclose(leg_block, soa_block, atol=1e-12), (
            f"Dense block {block_names[i - 2]} mismatch"
        )

def test_get_all_interactions_sparse_parity(mock_contact_data):
    parts, holes, contacts = mock_contact_data
    leg_res = legacy_cc.get_all_interactions(parts, holes, contacts, sparse=False)
    soa_res = soa_cc.get_all_interactions(parts, holes, contacts, sparse=False)
    
    soa_pppp_dense = soa_res[0].to_dense().detach().cpu().numpy()
    soa_ppph_dense = soa_res[1].to_dense().detach().cpu().numpy()
    
    assert np.allclose(leg_res[0], soa_pppp_dense, atol=1e-12)
    assert np.allclose(leg_res[1], soa_ppph_dense, atol=1e-12)

def test_get_all_interactions_antisymmetry_signs(mock_contact_data):
    """
    Verifies that sign_ket and sign_bra (h-p vs p-h) are handled 
    consistently with the legacy ground truth.
    """
    parts, holes, _ = mock_contact_data

    # Contact 1: (p,h) | (p,h) 
    # Contact 2: (h,p) | (p,h)
    special_contacts = [
        [parts[0], holes[0], parts[1], holes[1], 1.5],
        [holes[0], parts[0], parts[1], holes[1], 1.5], 
    ]

    leg_res = legacy_cc.get_all_interactions(parts, holes, special_contacts, sparse=False)
    soa_res = soa_cc.get_all_interactions(parts, holes, special_contacts, sparse=False)

    
    leg_vphph = leg_res[3]
    soa_vphph = soa_res[3].detach().cpu().numpy()

    assert np.allclose(leg_vphph, soa_vphph, atol=1e-12), \
        "SoA v_phph does not match Legacy ground truth distribution"
    
    assert np.isclose(np.sum(leg_vphph), np.sum(soa_vphph), atol=1e-12)
