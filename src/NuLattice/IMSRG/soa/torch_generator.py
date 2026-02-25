# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to construct IMSRG generator using PyTorch."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import torch

def get_hole_spes(occs: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Extracts single-particle energies for hole states using PyTorch masking.
    """
    return occs * torch.diag(f)


def get_particle_spes(occs: torch.Tensor, f: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """
    Extracts single-particle energies for particle states using PyTorch masking.
    """
    return (1 - occs) * (torch.diag(f) + delta)


def build_1b_energy_difference(occs: torch.Tensor, f: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """
    Constructs one-body energy differences (eps_i - eps_a).
    Uses broadcasting to avoid O(N^2) loops.
    """
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    # (N, 1) - (1, N) -> (N, N)
    # (e_i - e_a) in the ia block
    f_hp = spe_h[:, None] - spe_p[None, :]

    # Antisymmetrize to get (e_a - e_i) in the ai block
    # 1e-20 prevents division by zero during generator construction
    return f_hp - f_hp.T + 1e-20


def build_2b_energy_difference(occs: torch.Tensor, f: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """
    Constructs two-body energy differences (eps_i + eps_j - eps_a - eps_b).
    Uses 4D broadcasting to avoid O(N^4) contractions.
    """
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    # (N, N, N, N)
    # Calculates (e_i + e_j) - (e_a + e_b) efficiently
    gamma_hhpp = (
        spe_h[:, None, None, None] + spe_h[None, :, None, None] 
        - spe_p[None, None, :, None] - spe_p[None, None, None, :]
    )

    # Antisymmetrize to fill the pphh block
    return gamma_hhpp - gamma_hhpp.permute(2, 3, 0, 1) + 1e-20


def build_1b_arctan_generator(occs: torch.Tensor, f: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """
    Constructs the 1-body arctan generator using boolean indexing.
    """
    e_diff = build_1b_energy_difference(occs, f, delta)

    h = (occs > 0.5)
    p = ~h

    hp_mask = (h[:, None] & p[None, :]) | (p[:, None] & h[None, :])

    eta = torch.zeros_like(f)
    
    eta[hp_mask] = 0.5 * torch.arctan(2 * f[hp_mask] / e_diff[hp_mask])

    return eta


def build_2b_arctan_generator(occs: torch.Tensor, f: torch.Tensor, gamma: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """
    Constructs the 2-body arctan generator using boolean indexing.
    """
    e_diff = build_2b_energy_difference(occs, f, delta)

    h = (occs > 0.5)
    p = ~h

    hhpp_mask = (
        h[:, None, None, None] & h[None, :, None, None] & 
        p[None, None, :, None] & p[None, None, None, :]
    )

    pphh_mask = hhpp_mask.permute(2, 3, 0, 1)

    mask = hhpp_mask | pphh_mask

    eta = torch.zeros_like(gamma)
    
    eta[mask] = 0.5 * torch.arctan(2 * gamma[mask] / e_diff[mask])

    return eta
