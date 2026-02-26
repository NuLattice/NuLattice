import pytest
import torch
import numpy as np
from NuLattice._types import TwoBodyOperator

import NuLattice.CCM.coupled_cluster as legacy_cc
import NuLattice.CCM.soa.coupled_cluster as soa_cc

@pytest.fixture
def physics_params():
    """Standard physical parameters for a small lattice test."""
    return {
        "thisL": 2,          
        "holes": [[0,0,0,0,0], [0,0,0,1,0]], # 2 neutrons at origin (sz=0, tz=0/1)
        "vT1": -1.0,
        "vS1": -0.5,
        "str_3NF": 0.1
    }


def test_get_norm_ord_int_parity(physics_params):
    p = physics_params
    
    vacEn_leg, f_leg, v_leg = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], 
        str_3NF=p["str_3NF"], sparse=False
    )

    vacEn_soa, f_soa, v_soa = soa_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], 
        str_3NF=p["str_3NF"], sparse=True
    )

    assert np.isclose(vacEn_leg, vacEn_soa.item(), atol=1e-12)
    for i in range(3):
        assert np.allclose(f_leg[i], f_soa[i].detach().cpu().numpy(), atol=1e-12)

    pnum, hnum = f_leg[1].shape 
    
    expected_shapes = {
        0: (pnum, pnum, pnum, pnum), 
        1: (pnum, pnum, pnum, hnum), 
        2: (pnum, pnum, hnum, hnum), 
        3: (pnum, hnum, pnum, hnum), 
        4: (pnum, hnum, hnum, hnum), 
        5: (hnum, hnum, hnum, hnum), 
    }

    for i in range(6):
        leg_val = v_leg[i]
        
        if isinstance(v_soa[i], TwoBodyOperator):
            soa_dense = v_soa[i].to_dense().detach().cpu().numpy()
            target_shape = expected_shapes[i]
            soa_val = soa_dense[:target_shape[0], :target_shape[1], :target_shape[2], :target_shape[3]]
        else:
            soa_val = v_soa[i].detach().cpu().numpy()

        assert np.allclose(leg_val, soa_val, atol=1e-12), f"Block {i} shape {soa_val.shape} vs {leg_val.shape} failed parity"

def test_get_norm_ord_int_no_3nf_parity(physics_params):
    """Verifies parity when str_3NF is 0 (base 2-body case)."""
    p = physics_params
    vacEn_leg, f_leg, v_leg = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], 
        str_3NF=0, sparse=False
    )

    vacEn_soa, f_soa, v_soa = soa_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], 
        str_3NF=0, sparse=False, device=torch.device("cpu")
    )

    assert np.isclose(vacEn_leg, vacEn_soa.item(), atol=1e-12)
    for i in range(3):
        assert np.allclose(f_leg[i], f_soa[i].detach().numpy(), atol=1e-12)
