# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module defining normal ordering utilities for the IMSRG."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import numpy as np


def create_occupations(basis, ref):
    """
    Creates an array of occupation numbers indicating which states in the basis are occupied in the reference state

    :param basis:   List of all possible single-particle states in the basis
    :type basis:    list[tuple[int, int, int, int, int]]
    :param ref:     List of occupied single-particle states in the reference configuration
    :type ref:      list[tuple[int, int, int, int, int]]
    :return:        Array with 1.0 for occupied states and 0.0 for unoccupied states
    :rtype:         numpy array
    """
    occs = np.zeros(shape=(len(basis)))

    for x in ref:
        i = basis.index(x)
        occs[i] = 1.0

    return occs


def expand_h2(h2):
    """
    Expands two-body matrix elements by applying antisymmetrization relations

    Generates all antisymmetrized matrix elements from the input two-body interactions
    using the relations <pq|rs> = -<qp|rs> = -<pq|sr> = <qp|sr>

    :param h2:      List of two-body matrix elements in format [p, q, r, s, matrix_element]
    :type h2:       list[tuple[int, int, int, int, float]]
    :return:        Expanded list with all antisymmetrized two-body matrix elements
    :rtype:         list[tuple[int, int, int, int, float]]
    """
    h2_new = []
    for pp, qq, rr, ss, mme in h2:
        for p, q, r, s, me in [
            (pp, qq, rr, ss, mme),
            (qq, pp, rr, ss, -1 * mme),
            (pp, qq, ss, rr, -1 * mme),
            (qq, pp, ss, rr, mme),
        ]:
            h2_new.append((p, q, r, s, me))

    return h2_new


def get_three_body_permutations(pp, qq, rr):
    """
    Generates all antisymmetrized permutations of three indices with appropriate signs

    Computes the six permutations of three indices with the correct antisymmetrization
    factors for fermionic systems

    :param pp:      First index
    :type pp:       int
    :param qq:      Second index
    :type qq:       int
    :param rr:      Third index
    :type rr:       int
    :return:        List of permutations with format [index1, index2, index3, sign_factor]
    :rtype:         list[tuple[int, int, int, float]]
    """
    return [
        (pp, qq, rr, 1.0),
        (rr, pp, qq, 1.0),
        (qq, rr, pp, 1.0),
        (qq, pp, rr, -1.0),
        (pp, rr, qq, -1.0),
        (rr, qq, pp, -1.0),
    ]


def expand_h3(h3):
    """
    Expands three-body matrix elements by applying antisymmetrization relations

    Generates all antisymmetrized three-body matrix elements from the input interactions
    (with p < q < r and s < t < u) by applying permutations to both bra and ket indices with appropriate sign factors

    :param h3:      List of three-body matrix elements in format [p, q, r, s, t, u, matrix_element]
    :type h3:       list[tuple[int, int, int, int, int, int, float]]
    :return:        Expanded list with all antisymmetrized three-body matrix elements
    :rtype:         list[tuple[int, int, int, int, int, int, float]]
    """
    h3_new = []
    for pp, qq, rr, ss, tt, uu, mme in h3:
        for p, q, r, factor_pqr in get_three_body_permutations(pp, qq, rr):
            for s, t, u, factor_stu in get_three_body_permutations(ss, tt, uu):
                h3_new.append((p, q, r, s, t, u, factor_pqr * factor_stu * mme))

    return h3_new


def compute_normal_ordered_hamiltonian_no2b(occs, h1, h2, h3=None):
    """
    Computes the normal-ordered Hamiltonian with respect to a reference state

    Transforms the Hamiltonian to normal-ordered form by summing over occupied states,
    yielding the reference-state energy, effective one-body (Fock) operator, and effective two-body interactions

    :param occs:    Occupation numbers for each single-particle state (1.0 if occupied, 0.0 if empty)
    :type occs:     numpy array
    :param h1:      One-body matrix elements in format [p, q, matrix_element]
    :type h1:       list[tuple[int, int, float]]
    :param h2:      Two-body matrix elements in format [p, q, r, s, matrix_element]
    :type h2:       list[tuple[int, int, int, int, float]]
    :param h3:      Optional; three-body matrix elements in format [p, q, r, s, t, u, matrix_element]
    :type h3:       list[tuple[int, int, int, int, int, int, float]] | None
    :return:        Reference state energy, normal-ordered one-body operator (Fock matrix), normal-ordered two-body operator
    :rtype:         float, numpy array, numpy array
    """
    dim = len(occs)
    e0 = 0.0
    f = np.zeros((dim, dim))
    gamma = np.zeros((dim, dim, dim, dim))

    for p, q, me in h1:
        if p == q:
            e0 += occs[p] * me
        f[p, q] += me

    h2 = expand_h2(h2)
    for p, q, r, s, me in h2:
        if q == s:
            if p == r:
                e0 += occs[p] * occs[q] * 0.5 * me
            f[p, r] += occs[q] * me
        gamma[p, q, r, s] += me

    if h3 is not None:
        h3 = expand_h3(h3)
        for p, q, r, s, t, u, me in h3:
            if r == u:
                if q == t:
                    if p == s:
                        e0 += occs[p] * occs[q] * occs[r] * (1 / 6) * me
                    f[p, s] += occs[q] * occs[r] * 0.5 * me
                gamma[p, q, s, t] += me

    return e0, f, gamma
