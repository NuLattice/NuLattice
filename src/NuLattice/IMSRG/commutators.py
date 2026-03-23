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
import numpy as np

def antisymmetrize_2b_pq(a2: np.ndarray) -> np.ndarray:
    """
    Antisymmetrizes a two-body operator with respect to the first two indices (pq)

    Applies the antisymmetrization A_{pq} = 1/2(1 - P_{pq}) where P_{pq} exchanges
    indices p and q

    :param a2:      Two-body matrix elements with indices pqrs
    :type a2:       numpy array
    :return:        Partially antisymmetrized two-body matrix elements
    :rtype:         numpy array
    """
    return _antisymmetrize_2b_pq_np(a2)

def _antisymmetrize_2b_pq_original(a2):
    return 0.5 * (a2 - opt_einsum.contract("pqrs->qprs", a2))

def _antisymmetrize_2b_pq_np(a2):
    return 0.5 * (a2 - np.swapaxes(a2, 0, 1))


def antisymmetrize_2b_rs(a2: np.ndarray) -> np.ndarray:
    """
    Antisymmetrizes a two-body operator with respect to the last two indices (rs)

    Applies the antisymmetrization A_{rs} = 1/2(1 - P_{rs}) where P_{rs} exchanges
    indices r and s

    :param a2:      Two-body matrix elements with indices pqrs
    :type a2:       numpy array
    :return:        Partially antisymmetrized two-body matrix elements
    :rtype:         numpy array
    """
    return _antisymmetrize_2b_rs_np(a2)

def _antisymmetrize_2b_rs_original(a2):
    return 0.5 * (a2 - opt_einsum.contract("pqrs->pqsr", a2))

def _antisymmetrize_2b_rs_np(a2):
    return 0.5 * (a2 - np.swapaxes(a2, 2, 3))


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
    return _evaluate_comm_221_np(occs, a2, b2)

def _evaluate_comm_221_original(occs, a2, b2):
    # This version is a factor of 3 faster than the naive
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

# NOTE(vivek): Optimization: Reshaping for 2D GEMM.
# Logic: The contraction 'irpq, pqjr -> ij' is an O(N^4) operation.
# By combining 'p,q' --> standard matrix-matrix mul
def _evaluate_comm_221_np(occs, a2, b2):
    dim = len(occs)
    occsbar = 1 - occs

    weights = (
        (occsbar[None, None, :, None] * occsbar[None, None, None, :] * occs[None, :, None, None]) + 
        (occs[None, None, :, None] * occs[None, None, None, :] * occsbar[None, :, None, None])
    )

    a_weighted = a2 * weights
    b_weighted = b2 * weights

    # pqjr -> rpqj same as transpose(3, 0, 1, 2)
    atm = a2.transpose(3, 0, 1, 2).reshape(-1, dim)
    btm = b2.transpose(3, 0, 1, 2).reshape(-1, dim)

    # (i, r*p*q) & (r*p*q, j) -> (i, j)
    res = 0.5 * (
        np.matmul(a_weighted.reshape(dim, -1), btm) - 
        np.matmul(b_weighted.reshape(dim, -1), atm)
    )
    
    return res

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
    return _evaluate_comm_222_pphh_np(occs, a2, b2)

def _evaluate_comm_222_pphh_original(occs, a2, b2):
    
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

# NOTE(vivek): reshaping rank 4 to 2
# Logic: This is an O(N^6) bottleneck. Reshaping (N,N,N,N) into (N^2, N^2) 
def _evaluate_comm_222_pphh_np(occs, a2, b2):
    dim = len(occs)
    occsbar = 1 - occs

    # Weighting factors applied via broadcasting
    # (n-n) term from IMSRG flow
    q_factor = (occsbar[:, None] * occsbar[None, :]) - (occs[:, None] * occs[None, :])
    

    # reshape to (N**2, N**2)
    A = a2.reshape(dim**2, dim**2)
    B = b2.reshape(dim**2, dim**2)
    A_weighted = (a2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)    
    B_weighted = (b2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)    

    res = 0.5 * (np.matmul(A_weighted, B) - np.matmul(B_weighted, A))
    return res.reshape(dim, dim, dim, dim)

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
    return _evaluate_comm_222_ph_np(occs, a2, b2)

def _evaluate_comm_222_ph_original(occs, a2, b2):
    # This is faster than naive version above because we only do half as many BLAS operations
    occsbar = 1 - occs

    a2_with_occs2 = opt_einsum.contract(
        "p,q,pjkq->pjkq", occs, occsbar, a2
    ) - opt_einsum.contract("p,q,pjkq->pjkq", occsbar, occs, a2)

    return antisymmetrize_2b(
        -4
        * opt_einsum.contract("pjkq,iqpl->ijkl", a2_with_occs2, b2, optimize="greedy")
    )


# NOTE(vivek): Transpose + Reshape + GEMM.
# Logic: Handle the cross-contraction (particle-hole) by reordering indices 
# creates adjacent contracted indices
def _evaluate_comm_222_ph_np(occs, a2, b2):
    dim = len(occs)
    occsbar = 1 - occs
    
    # n_p * (1 - n_q) - (1 - n_p) * n_q
    ph_factor = (occs[:, None] * occsbar[None, :]) - (occsbar[:, None] * occs[None, :])
    
    # contraction pjkq, iqpl -> ijkl
    # i, j, k, l -> j, k, p, q (1, 2, 0, 3)
    # i, j, k, l -> i, l, p, q (0, 3, 2, 1)
    # align q and p indices

    A_weighted = (a2 * ph_factor[:, None, None, :]).transpose(1, 2, 0, 3).reshape(dim**2, dim**2)    
    B = b2.transpose(2, 1, 0, 3).reshape(dim**2, dim**2)

    c_mat = np.matmul(A_weighted, B)
    
    res = -4 * c_mat.reshape(dim, dim, dim, dim).transpose(2, 0, 1, 3)
    return antisymmetrize_2b(res)

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

# NOTE(vivek): this is the only "public" function called by imsrg_rhs in the ode solver. If this is running multiple times,
# there is a lot of waste here in having to recompute factors, weights, and certain reshapes which trigger garbage collection.
# On CPU, function calls are effectively negligible compared to the functions themselves, but on GPU, these become 9 kernel calls
# which is not a fun time
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

def _evaluate_imsrg_commutator_np(occs, a1, a2, b1, b2):

    dim = len(occs)
    occsbar = 1 - occs
    np_nbarq = occs[:, None] * occsbar[None, :]
    nbarp_nq = occs[None, :] * occsbar[:, None]
    q_factor = (occsbar[:, None] * occsbar[None, :]) - (occs[:, None] * occs[None, :])
    ph_factor = (occs[:, None] * occsbar[None, :]) - (occsbar[:, None] * occs[None, :])
 

    # [1,1]->0 and [2,2]->0
    res0 = 0
    res0 += np.sum(np_nbarq * a1 * b1.T) - np.sum(nbarp_nq * a1 * b1.T)
    term220 = np.einsum("p,q,r,s,pqrs,rspq", occs, occs, occsbar, occsbar, a2, b2)
    term220 -= np.einsum("p,q,r,s,pqrs,rspq", occsbar, occsbar, occs, occs, a2, b2)
    res0 += 0.25 * term220

    res1 = (a1 @ b1) - (b1 @ a1)
    
    # [1,2]->1:
    # We combine (a1*b2 - b1*a2) weighted by ph_weight
    a1_b2_fused = np.einsum("pq,iqjp->ij", ph_factor * a1, b2)
    b1_a2_fused = np.einsum("pq,iqjp->ij", ph_factor * b1, a2)
    res1 += (a1_b2_fused - b1_a2_fused)
    
    # [2,2]->1
    # (nbar_p nbar_q n_r) + (n_p n_q nbar_r)
    w_221 = (occsbar[None, None, :, None] * occsbar[None, None, None, :] * occs[None, :, None, None]) + \
            (occs[None, None, :, None] * occs[None, None, None, :] * occsbar[None, :, None, None])
    
    a2_w = (a2 * w_221).reshape(dim, -1)
    b2_w = (b2 * w_221).reshape(dim, -1)
    # rpqj -> transpose(3, 0, 1, 2)
    a2_tr = a2.transpose(3, 0, 1, 2).reshape(-1, dim)
    b2_tr = b2.transpose(3, 0, 1, 2).reshape(-1, dim)
    
    res1 += 0.5 * (a2_w @ b2_tr - b2_w @ a2_tr)

    diff122 = (a1 @ b2.reshape(dim, -1)).reshape(dim, dim, dim, dim)
    diff122 -= (np.einsum("pk,ijpl->ijkl", a1, b2)) # Standard scaling
    diff122 -= (b1 @ a2.reshape(dim, -1)).reshape(dim, dim, dim, dim)
    diff122 += (np.einsum("pk,ijpl->ijkl", b1, a2))
    
    # Single antisymmetrization call for all 122 terms
    _a2 = 2.0 * diff122
    _2b_pq = 0.5 * (_a2 - np.swapaxes(_a2, 0, 1))
    res2 = 0.5 * (_2b_pq - np.swapaxes(_2b_pq, 2, 3))

    # [2,2]->2 pphh
    a2_mat = a2.reshape(dim**2, dim**2)
    b2_mat = b2.reshape(dim**2, dim**2)
    a2_q = (a2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)
    b2_q = (b2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)
    
    res2 += 0.5 * (a2_q @ b2_mat - b2_q @ a2_mat).reshape(dim, dim, dim, dim)

    # [2,2]->2 ph: GEMM Optimized
    # Align to (j,k,p,q) and (p,q,i,l) for matmul
    a2_ph = (a2 * ph_factor[:, None, None, :]).transpose(1, 2, 0, 3).reshape(dim**2, dim**2)
    b2_ph_tr = b2.transpose(2, 1, 0, 3).reshape(dim**2, dim**2)
    
    c_ph = (a2_ph @ b2_ph_tr).reshape(dim, dim, dim, dim).transpose(2, 0, 1, 3)
    _a2 = -4.0 * c_ph
    _2b_pq = 0.5 * (_a2 - np.swapaxes(_a2, 0, 1))
    res2 += 0.5 * (_2b_pq - np.swapaxes(_2b_pq, 2, 3))

    return res0, res1, res2
