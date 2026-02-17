# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to evaluate the commutators of the IMSRG."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import opt_einsum


def antisymmetrize_2b_pq(a2):
    """
    Antisymmetrizes a two-body operator with respect to the first two indices (pq)

    Applies the antisymmetrization A_{pq} = 1/2(1 - P_{pq}) where P_{pq} exchanges
    indices p and q

    :param a2:      Two-body matrix elements with indices pqrs
    :type a2:       numpy array
    :return:        Partially antisymmetrized two-body matrix elements
    :rtype:         numpy array
    """
    return 0.5 * (a2 - opt_einsum.contract("pqrs->qprs", a2))


def antisymmetrize_2b_rs(a2):
    """
    Antisymmetrizes a two-body operator with respect to the last two indices (rs)

    Applies the antisymmetrization A_{rs} = 1/2(1 - P_{rs}) where P_{rs} exchanges
    indices r and s

    :param a2:      Two-body matrix elements with indices pqrs
    :type a2:       numpy array
    :return:        Partially antisymmetrized two-body matrix elements
    :rtype:         numpy array
    """
    return 0.5 * (a2 - opt_einsum.contract("pqrs->pqsr", a2))


def antisymmetrize_2b(a2):
    """
    Fully antisymmetrizes a two-body operator with respect to both pairs of indices

    Applies complete antisymmetrization to both bra and ket indices, equivalent to
    A_{pq}A_{rs} acting on the operator

    :param a2:      Two-body matrix elements with indices pqrs
    :type a2:       numpy array
    :return:        Fully antisymmetrized two-body matrix elements
    :rtype:         numpy array
    """
    return antisymmetrize_2b_rs(antisymmetrize_2b_pq(a2))


def evaluate_comm_110(occs, a1, b1):
    """
    Evaluates the [1,1]->0 commutator contribution

    Computes the scalar (0-body) part of the commutator between two one-body operators

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a1:      First one-body operator
    :type a1:       numpy array
    :param b1:      Second one-body operator
    :type b1:       numpy array
    :return:        Scalar commutator contribution
    :rtype:         float
    """
    occsbar = 1 - occs

    val = 0.0
    val += opt_einsum.contract("p,q,pq,qp", occs, occsbar, a1, b1)
    val -= opt_einsum.contract("p,q,pq,qp", occsbar, occs, a1, b1)

    return val


def evaluate_comm_111(occs, a1, b1):
    """
    Evaluates the [1,1]->1 commutator contribution

    Computes the one-body part of the commutator between two one-body operators

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a1:      First one-body operator
    :type a1:       numpy array
    :param b1:      Second one-body operator
    :type b1:       numpy array
    :return:        One-body commutator contribution
    :rtype:         numpy array
    """
    occsbar = 1 - occs

    return opt_einsum.contract("ip,pj->ij", a1, b1) - opt_einsum.contract(
        "ip,pj->ij", b1, a1
    )


def evaluate_comm_121(occs, a1, b2):
    """
    Evaluates the [1,2]->1 commutator contribution

    Computes the one-body part of the commutator between a one-body and two-body operator

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a1:      One-body operator
    :type a1:       numpy array
    :param b2:      Two-body operator
    :type b2:       numpy array
    :return:        One-body commutator contribution
    :rtype:         numpy array
    """
    occsbar = 1 - occs

    return opt_einsum.contract(
        "p,q,pq,iqjp->ij", occs, occsbar, a1, b2
    ) - opt_einsum.contract("p,q,pq,iqjp->ij", occsbar, occs, a1, b2)


def evaluate_comm_122(occs, a1, b2):
    """
    Evaluates the [1,2]->2 commutator contribution

    Computes the two-body part of the commutator between a one-body and two-body operator

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a1:      One-body operator
    :type a1:       numpy array
    :param b2:      Two-body operator
    :type b2:       numpy array
    :return:        Two-body commutator contribution
    :rtype:         numpy array
    """
    occsbar = 1 - occs

    return antisymmetrize_2b(
        2
        * (
            opt_einsum.contract("ip,pjkl->ijkl", a1, b2)
            - opt_einsum.contract("pk,ijpl->ijkl", a1, b2)
        )
    )


def evaluate_comm_220(occs, a2, b2):
    """
    Evaluates the [2,2]->0 commutator contribution

    Computes the scalar (0-body) part of the commutator between two two-body operators

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        Scalar commutator contribution
    :rtype:         float
    """
    occsbar = 1 - occs

    val = 0.0

    val += opt_einsum.contract(
        "p,q,r,s,pqrs,rspq", occs, occs, occsbar, occsbar, a2, b2
    )
    val -= opt_einsum.contract(
        "p,q,r,s,pqrs,rspq", occsbar, occsbar, occs, occs, a2, b2
    )

    return 0.25 * val


def __evaluate_comm_221_naive(occs, a2, b2):
    """
    Evaluates the [2,2]->1 commutator contribution using naive implementation

    Computes the one-body part of the commutator between two two-body operators
    using a straightforward but potentially less efficient contraction pattern

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        One-body commutator contribution
    :rtype:         numpy array
    """
    occsbar = 1 - occs

    return 0.5 * (
        opt_einsum.contract("p,q,r,irpq,pqjr->ij", occsbar, occsbar, occs, a2, b2)
        + opt_einsum.contract("p,q,r,irpq,pqjr->ij", occs, occs, occsbar, a2, b2)
        - opt_einsum.contract("p,q,r,irpq,pqjr->ij", occsbar, occsbar, occs, b2, a2)
        - opt_einsum.contract("p,q,r,irpq,pqjr->ij", occs, occs, occsbar, b2, a2)
    )


def evaluate_comm_221(occs, a2, b2):
    """
    Evaluates the [2,2]->1 commutator contribution using optimized implementation

    Computes the one-body part of the commutator between two two-body operators.
    This implementation pre-computes tensors contracted with occupation numbers
    for improved efficiency

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        One-body commutator contribution
    :rtype:         numpy array
    """
    # This version is a factor of 3 faster than the version above
    # Half comes from combining occupations
    # The other half comes from massaging things into a form where opt_einsum performs a BLAS GEMM
    # rather than a tensor_dot TDOT
    occsbar = 1 - occs

    a2_with_occs = opt_einsum.contract(
        "p,q,r,irpq->irpq", occsbar, occsbar, occs, a2
    ) + opt_einsum.contract("p,q,r,irpq->irpq", occs, occs, occsbar, a2)
    b2_with_occs = opt_einsum.contract(
        "p,q,r,irpq->irpq", occsbar, occsbar, occs, b2
    ) + opt_einsum.contract("p,q,r,irpq->irpq", occs, occs, occsbar, b2)
    a2_trans = opt_einsum.contract("pqjr->rpqj", a2)
    b2_trans = opt_einsum.contract("pqjr->rpqj", b2)

    return 0.5 * (
        opt_einsum.contract("irpq,rpqj->ij", a2_with_occs, b2_trans, optimize="greedy")
        - opt_einsum.contract("irpq,rpqj->ij", b2_with_occs, a2_trans, optimize="greedy")
    )


def __evaluate_comm_222_naive(occs, a2, b2):
    """
    Evaluates the [2,2]->2 commutator contribution using naive implementation

    Computes the two-body part of the commutator between two two-body operators
    using a straightforward contraction approach that may be less computationally efficient

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        Two-body commutator contribution
    :rtype:         numpy array
    """
    occsbar = 1 - occs

    return 0.5 * (
        opt_einsum.contract("p,q,ijpq,pqkl->ijkl", occsbar, occsbar, a2, b2, optimize="greedy")
        - opt_einsum.contract("p,q,ijpq,pqkl->ijkl", occs, occs, a2, b2, optimize="greedy")
        - opt_einsum.contract("p,q,ijpq,pqkl->ijkl", occsbar, occsbar, b2, a2, optimize="greedy")
        + opt_einsum.contract("p,q,ijpq,pqkl->ijkl", occs, occs, b2, a2, optimize="greedy")
    ) + antisymmetrize_2b(
        -4
        * (
            opt_einsum.contract("p,q,pjkq,iqpl->ijkl", occs, occsbar, a2, b2, optimize="greedy")
            - opt_einsum.contract("p,q,pjkq,iqpl->ijkl", occsbar, occs, a2, b2, optimize="greedy")
        )
    )


def evaluate_comm_222_pphh(occs, a2, b2):
    """
    Evaluates the particle-particle hole-hole contribution to the [2,2]->2 commutator

    Computes the specific part of the two-body commutator involving contractions
    between particle-particle and hole-hole index pairs

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        Particle-particle hole-hole commutator contribution
    :rtype:         numpy array
    """
    # This is faster than naive version above because we only do half as many BLAS operations
    occsbar = 1 - occs

    a2_with_occs = opt_einsum.contract(
        "p,q,ijpq->ijpq", occsbar, occsbar, a2
    ) - opt_einsum.contract("p,q,ijpq->ijpq", occs, occs, a2)
    b2_with_occs = opt_einsum.contract(
        "p,q,ijpq->ijpq", occsbar, occsbar, b2
    ) - opt_einsum.contract("p,q,ijpq->ijpq", occs, occs, b2)

    return 0.5 * (
        opt_einsum.contract("ijpq,pqkl->ijkl", a2_with_occs, b2, optimize="greedy")
        - opt_einsum.contract("ijpq,pqkl->ijkl", b2_with_occs, a2, optimize="greedy")
    )


def evaluate_comm_222_ph(occs, a2, b2):
    """
    Evaluates the particle-hole contribution to the [2,2]->2 commutator

    Computes the specific part of the two-body commutator involving contractions
    between particle-hole index pairs with proper antisymmetrization

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        Particle-hole commutator contribution
    :rtype:         numpy array
    """
    # This is faster than naive version above because we only do half as many BLAS operations
    occsbar = 1 - occs

    a2_with_occs2 = opt_einsum.contract(
        "p,q,pjkq->pjkq", occs, occsbar, a2
    ) - opt_einsum.contract("p,q,pjkq->pjkq", occsbar, occs, a2)

    return antisymmetrize_2b(
        -4
        * opt_einsum.contract("pjkq,iqpl->ijkl", a2_with_occs2, b2, optimize="greedy")
    )


def evaluate_comm_222(occs, a2, b2):
    """
    Evaluates the complete [2,2]->2 commutator contribution

    Computes the full two-body part of the commutator between two two-body operators
    by combining particle-particle hole-hole and particle-hole contributions

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a2:      First two-body operator
    :type a2:       numpy array
    :param b2:      Second two-body operator
    :type b2:       numpy array
    :return:        Complete two-body commutator contribution
    :rtype:         numpy array
    """
    return evaluate_comm_222_pphh(occs, a2, b2) + evaluate_comm_222_ph(occs, a2, b2)


def evaluate_imsrg2_commutator(occs, a1, a2, b1, b2):
    """
    Evaluates the complete commutator for IMSRG(2) flow equations

    Computes all IMSRG(2) contributions to the commutator C = [A, B]
    where A and B each contain one- and two-body parts,
    returning the 0-body, 1-body, and 2-body contributions to the result C

    :param occs:    Occupation numbers for each single-particle state
    :type occs:     numpy array
    :param a1:      One-body part of first operator
    :type a1:       numpy array
    :param a2:      Two-body part of first operator
    :type a2:       numpy array
    :param b1:      One-body part of second operator
    :type b1:       numpy array
    :param b2:      Two-body part of second operator
    :type b2:       numpy array
    :return:        Zero-body, one-body, and two-body commutator contributions
    :rtype:         float, numpy array, numpy array
    """

    res0 = evaluate_comm_110(occs, a1, b1) + evaluate_comm_220(occs, a2, b2)
    res1 = (
        evaluate_comm_111(occs, a1, b1)
        + evaluate_comm_121(occs, a1, b2)
        - evaluate_comm_121(occs, b1, a2)
        + evaluate_comm_221(occs, a2, b2)
    )
    res2 = (
        evaluate_comm_122(occs, a1, b2)
        - evaluate_comm_122(occs, b1, a2)
        + evaluate_comm_222(occs, a2, b2)
    )

    return res0, res1, res2
