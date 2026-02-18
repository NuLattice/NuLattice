# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to construct IMSRG generator."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import numpy as np
import opt_einsum


def get_hole_spes(occs: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Extracts single-particle energies for hole states from the Fock matrix

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       Fock matrix containing single-particle energies on the diagonal
    :type f:        numpy array
    :return:        Single-particle energies for occupied (hole) states
    :rtype:         numpy array
    """
    return np.multiply(occs, np.diag(f))


def get_particle_spes(occs: np.ndarray, f: np.ndarray, delta: float =0.0):
    """
    Extracts single-particle energies for particle states from the Fock matrix

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       Fock matrix containing single-particle energies on the diagonal
    :type f:        numpy array
    :param delta:   Optional; energy shift to avoid degeneracies in denominators
    :type delta:    float
    :return:        Single-particle energies for unoccupied (particle) states
    :rtype:         numpy array
    """
    return np.multiply(1 - occs, np.diag(f) + delta)


def build_1b_energy_difference(occs: np.ndarray, f: np.ndarray, delta: float =0.0) -> np.ndarray:
    """
    Constructs one-body energy differences for IMSRG generator denominators

    Computes the energy differences e_i - e_a between hole and particle states,
    which appear in the denominators of the IMSRG generator. A small regularization
    term prevents division by zero

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       Fock matrix containing single-particle energies
    :type f:        numpy array
    :param delta:   Optional; energy shift to avoid degeneracies in denominators
    :type delta:    float
    :return:        Matrix of energy differences for one-body generator denominators
    :rtype:         numpy array
    """
    return _build_1b_energy_difference_original(occs, f, delta)

def _build_1b_energy_difference_original(occs: np.ndarray, f: np.ndarray, delta: float =0.0):
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    ones = np.ones_like(spe_h)

    # e_i - e_a
    f_hp = opt_einsum.contract("i,a->ia", spe_h, ones) - opt_einsum.contract(
        "i,a->ia", ones, spe_p
    )

    # 1e-20 prevents division by 0 when we invert the energy differences to produce energy denominators
    return f_hp - opt_einsum.contract("ia->ai", f_hp) + 1e-20

def _build_1b_energy_difference_np(occs: np.ndarray, f: np.ndarray, delta: float =0.0):
    spe_h = np.diag(f) * occs
    spe_p = (np.diag(f) + delta) * (1 - occs)
    
    # (N, 1) - (1, N) -> (N, N)
    f_hp = spe_h[:, None] - spe_p[None, :]
    
    return f_hp - f_hp.T + 1e-20

def build_2b_energy_difference(occs: np.ndarray, f: np.ndarray, delta: float =0.0) -> np.ndarray:
    """
    Constructs two-body energy differences for IMSRG generator denominators

    Computes the energy differences e_i + e_j - e_a - e_b between hole and particle state pairs,
    which appear in the denominators of the IMSRG generator. A small regularization
    term prevents division by zero

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       Fock matrix containing single-particle energies
    :type f:        numpy array
    :param delta:   Optional; energy shift to avoid degeneracies in denominators
    :type delta:    float
    :return:        Tensor of energy differences for two-body generator denominators
    :rtype:         numpy array
    """
    return _build_2b_energy_difference_original(occs, f, delta)


# NOTE(vivek): nested broadcasting instead of 4+1 contractions
# single pass broadcast instead of 4 O(N**4) contractions
def _build_2b_energy_difference_np(occs: np.ndarray, f: np.ndarray, delta: float =0.0):
    spe_h = np.diag(f) * occs
    spe_p = (np.diag(f) + delta) * (1 - occs)

    gamma_hhpp = (spe_h[:, None, None, None] + spe_h[None, :, None, None] - 
                  spe_p[None, None, :, None] - spe_p[None, None, None, :])

    return gamma_hhpp - gamma_hhpp.transpose(2, 3, 0, 1) + 1e-20


def _build_2b_energy_difference_original(occs: np.ndarray, f: np.ndarray, delta: float =0.0):
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    ones = np.ones_like(spe_h)

    # e_i + e_j - e_a - e_b
    gamma_hhpp = (
        opt_einsum.contract("i,j,a,b->ijab", spe_h, ones, ones, ones)
        + opt_einsum.contract("i,j,a,b->ijab", ones, spe_h, ones, ones)
        - opt_einsum.contract("i,j,a,b->ijab", ones, ones, spe_p, ones)
        - opt_einsum.contract("i,j,a,b->ijab", ones, ones, ones, spe_p)
    )

    # 1e-20 prevents division by 0 when we invert the energy differences to produce energy denominators
    return gamma_hhpp - opt_einsum.contract("ijab->abij", gamma_hhpp) + 1e-20


def build_1b_arctan_generator(occs: np.ndarray, f: np.ndarray, delta: float =0.0) -> np.ndarray:
    """
    Constructs the one-body part of the arctan IMSRG generator

    Builds the generator eta^{(1)} = (1/2) arctan(2f^{ai}/e_{ai}) that drives
    the one-body part of the IMSRG flow equations, ensuring decoupling of
    particle-hole excitations from the reference state. An optional delta pushes
    up the energies of the particle states to prevent vanishing energy denominators
    and ensure numerical stability. Most calculations should not need this

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       One-body Fock matrix
    :type f:        numpy array
    :param delta:   Optional; energy shift to avoid degeneracies in denominators
    :type delta:    float
    :return:        One-body arctangent generator matrix elements
    :rtype:         numpy array
    """
    return _build_1b_arctan_generator_np(occs, f, delta)

def _build_1b_arctan_generator_original(occs, f, delta=0.0):
    e_diff = build_1b_energy_difference(occs, f, delta)

    # Mask to isolate hp and ph matrix elements
    hp_mask = opt_einsum.contract("i,a->ia", occs, 1 - occs) + opt_einsum.contract(
        "i,a->ai", occs, 1 - occs
    )

    return 0.5 * np.arctan(
        np.multiply(
            np.multiply(
                np.power(e_diff, -1),
                2 * f,
            ),
            hp_mask,
        )
    )

def _build_1b_arctan_generator_np(occs, f, delta=0.0):
    e_diff = _build_1b_energy_difference_np(occs, f, delta)

    h = (occs > 0.5)
    p = (occs < 0.5)
    
    # Create mask for particle-hole (ia) and hole-particle (ai)
    hp_mask = (h[:, None] & p[None, :]) | (p[:, None] & h[None, :])

    eta = np.zeros_like(f)
    eta[hp_mask] = 0.5 * np.arctan(2 * f[hp_mask] / e_diff[hp_mask])
    
    return eta

def build_2b_arctan_generator(occs: np.ndarray, f: np.ndarray, gamma: np.ndarray, delta: float =0.0) -> np.ndarray:
    """
    Constructs the two-body part of the arctan IMSRG generator

    Builds the generator eta^{(2)} = (1/2) arctan(2Gamma^{abij}/e_{abij}) that drives
    the two-body part of the IMSRG flow equations, ensuring decoupling of
    particle-hole excitations from the reference state. An optional delta pushes
    up the energies of the particle states to prevent vanishing energy denominators
    and ensure numerical stability. Most calculations should not need this

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param f:       One-body Fock matrix for constructing energy denominators
    :type f:        numpy array
    :param gamma:   Two-body interaction matrix
    :type gamma:    numpy array
    :param delta:   Optional; energy shift to avoid degeneracies in denominators
    :type delta:    float
    :return:        Two-body arctangent generator matrix elements
    :rtype:         numpy array
    """
    return _build_2b_arctan_generator_np(occs, f, gamma, delta)

def  _build_2b_arctan_generator_original(occs, f, gamma, delta=0.0):
    e_diff = build_2b_energy_difference(occs, f, delta)

    # Mask to isolate hhpp and pphh matrix elements
    hhpp_mask = opt_einsum.contract(
        "i,j,a,b->ijab", occs, occs, 1 - occs, 1 - occs
    ) + opt_einsum.contract("i,j,a,b->abij", occs, occs, 1 - occs, 1 - occs)

    return 0.5 * np.arctan(
        np.multiply(
            np.multiply(
                np.power(e_diff, -1),
                2 * gamma,
            ),
            hhpp_mask,
        )
    )

def _build_2b_arctan_generator_np(occs, f, gamma, delta=0.0):
    e_diff = _build_2b_energy_difference_np(occs, f, delta)

    # 1 byte per bool vs 8 bytes per float
    h = (occs > 0.5)
    p = (occs < 0.5)
    
    # Identify hhpp and pphh sectors
    hhpp_mask = (h[:, None, None, None] & h[None, :, None, None] & 
                 p[None, None, :, None] & p[None, None, None, :])
    pphh_mask = hhpp_mask.transpose(2, 3, 0, 1)
    
    mask = hhpp_mask | pphh_mask
    
    eta = np.zeros_like(gamma)
    eta[mask] = 0.5 * np.arctan(2 * gamma[mask] / e_diff[mask])
    
    return eta
