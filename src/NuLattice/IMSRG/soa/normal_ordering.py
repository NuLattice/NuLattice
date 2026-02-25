# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module defining normal ordering utilities for the IMSRG."""

__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

from NuLattice._types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator

import numpy as np
from typing import Optional

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def decorator(f):
            return f

        if len(args) > 0 and callable(args[0]):
            return args[0]
        return decorator


@njit
def create_occupations_nstates(n_states, ref_indices):
    """
    Creates occupation array from a list of occupied state indices.

    :param n_stat: Total number of single-particle states (int)
    :param ref_indices: Array of indices that are occupied (np.ndarray)
    """
    occs = np.zeros(n_states, dtype=np.float64)
    for idx in ref_indices:
        occs[idx] = 1.0
    return occs


def create_occupations(basis, ref_indices):
    """
    Creates occupation array from a list of occupied state indices.

    :param n_stat: Total number of single-particle states (int)
    :param ref_indices: Array of indices that are occupied (np.ndarray)
    """
    matches = (basis[:, None, :] == ref_indices[None, :, :]).all(axis=2)
    return matches.any(axis=1).astype(np.float64)



@njit
def _compute_op1_kernel(h1, dim, occs, e0_arr, f):
    for p in range(dim):
        for q in range(dim):
            val = h1[p, q]
            f[p, q] += val
            if p == q:
                e0_arr[0] += occs[p] * val


@njit
def _accumulate_2b_contributions(p, q, r, s, val, occs, e0_arr, f, gamma):
    """
    Helper to accumulate a single permuted 2-body element into normal ordered operators.
    """
    # 2-Body (Gamma) Contribution
    gamma[p, q, r, s] += val

    # 1-Body (Fock) Contribution: Contraction over q=s
    if q == s:
        term = occs[q] * val
        f[p, r] += term

        # 0-Body (Energy) Contribution: Contraction over p=r
        if p == r:
            e0_arr[0] += 0.5 * occs[p] * term


@njit
def _compute_op2_kernel(indices, values, occs, e0_arr, f, gamma):
    """
    Numba kernel to process 2-body interactions in SoA format.
    Applies 4 antisymmetrization permutations on-the-fly.
    """
    n_elems = len(values)
    for i in range(n_elems):
        p, q, r, s = indices[i]
        val = values[i]

        # Apply antisymmetrization: <pq|rs>
        _accumulate_2b_contributions(p, q, r, s, val, occs, e0_arr, f, gamma)
        _accumulate_2b_contributions(q, p, r, s, -val, occs, e0_arr, f, gamma)
        _accumulate_2b_contributions(p, q, s, r, -val, occs, e0_arr, f, gamma)
        _accumulate_2b_contributions(q, p, s, r, val, occs, e0_arr, f, gamma)


@njit
def _compute_op3_kernel(indices, values, occs, e0_arr, f, gamma):
    """
    Numba kernel to process 3-body interactions in SoA format.
    Applies 36 antisymmetrization permutations on-the-fly and computes NO2B approximation.
    """
    n_elems = len(values)

    # Pre-computed permutations for 3 indices (0,1,2)
    # Rows: [p,q,r] indices, Sign
    perm_map = np.array(
        [
            [0, 1, 2], [2, 0, 1], [1, 2, 0],  # Even (+1)
            [1, 0, 2], [0, 2, 1], [2, 1, 0],  # Odd (-1)
        ],
        dtype=np.int8,
    )
    perm_signs = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    for i in range(n_elems):
        # Extract raw indices and value
        p_raw, q_raw, r_raw = indices[i, 0], indices[i, 1], indices[i, 2]
        s_raw, t_raw, u_raw = indices[i, 3], indices[i, 4], indices[i, 5]
        val_raw = values[i]

        bra_raw = np.array([p_raw, q_raw, r_raw])
        ket_raw = np.array([s_raw, t_raw, u_raw])

        # Iterate over 6 Bra permutations
        for bi in range(6):
            p = bra_raw[perm_map[bi, 0]]
            q = bra_raw[perm_map[bi, 1]]
            r = bra_raw[perm_map[bi, 2]]
            sign_bra = perm_signs[bi]

            # Iterate over 6 Ket permutations
            for ki in range(6):
                s = ket_raw[perm_map[ki, 0]]
                t = ket_raw[perm_map[ki, 1]]
                u = ket_raw[perm_map[ki, 2]]
                sign_ket = perm_signs[ki]

                val = val_raw * sign_bra * sign_ket

                # Normal Ordering Approximations

                # Check for contraction on the 3rd index (r == u)
                # This contributes to the effective 2-body operator
                if r == u:
                    gamma[p, q, s, t] += val

                    # Check for contraction on 2nd index (q == t)
                    # This contributes to the effective 1-body operator
                    if q == t:
                        term = 0.5 * occs[q] * occs[r] * val
                        f[p, s] += term

                        # Check for contraction on 1st index (p == s)
                        # This contributes to the scalar energy (E0)
                        if p == s:
                            # Factor: 1/6 total (1/3 * 1/2 from prev step)
                            e0_arr[0] += (1.0 / 3.0) * occs[p] * term


def compute_normal_ordered_hamiltonian_no2b(
    occs: np.ndarray,
    op1: OneBodyOperator | np.ndarray,
    op2: TwoBodyOperator,
    op3: Optional[ThreeBodyOperator] = None,
):
    """
    Computes the normal-ordered Hamiltonian with respect to a reference state

    Transforms the Hamiltonian to normal-ordered form by summing over occupied states,
    yielding the reference-state energy, effective one-body (Fock) operator, and effective two-body interactions

    :param occs:    Occupation numbers (dim, )
    :type occs:     numpy bool array
    :return:        Reference state energy, normal-ordered one-body operator (Fock matrix), normal-ordered two-body operator
    :rtype:         float, numpy array, numpy array
    """
    dim = len(occs)
    e0 = np.zeros(1, dtype=np.float64)
    f = np.zeros((dim, dim), dtype=np.float64)
    gamma = np.zeros((dim, dim, dim, dim), dtype=np.float64)

    h1 = op1.to_dense() if hasattr(op1, "to_dense") else op1
    _compute_op1_kernel(h1, dim, occs, e0, f)

    _compute_op2_kernel(op2.indices, op2.values, occs, e0, f, gamma)

    if op3 is not None:
        _compute_op3_kernel(op3.indices, op3.values, occs, e0, f, gamma)

    return e0[0], f, gamma
