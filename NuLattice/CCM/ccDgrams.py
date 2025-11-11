"""
Provides functions for all of the coupled cluster diagrams
"""
__authors__   =  ["Maxwell Rothman", "Ben Johnson"]
__credits__   =  ["Maxwell Rothman", "Ben Johnson"]
__copyright__ = "(c) Maxwell Rothman and Ben Johnson"
__license__   = "BSD-3-Clause"
__date__      = "2025-07-26"

import numpy as np
from opt_einsum import contract

def v_ppph_dgrams(v_ppph, t1, t2):
    """
    Calculates all 6 of the diagrams that use v_ppph using the fact that it is sparse

    :param v_ppph:  nonzeroes of the two body interaction matrix V^{ab}_{ck}
    :type v_ppph:   list[(int, int, int, int, float)]
    :param t1:      T1 matrix t_i^a
    :type t1:       numpy array
    :param t2:      T2 matrix t_{ij}^{ab}
    :type t2:       numpy array
    :return:        The result of the 6 diagrams that contribute to t1 and t2
    :rtype:         list[numpy array]
    """
    pnum = np.shape(t1)[0]
    hnum = np.shape(t1)[1]
    ret0 = np.zeros((pnum, hnum))
    ret1 = np.zeros((pnum, pnum))
    ret2 = np.zeros((pnum, pnum, hnum, hnum))
    ret3 = np.zeros((pnum, pnum))
    ret4 = np.zeros((pnum, pnum, hnum, hnum))
    ret5 = np.zeros((pnum, hnum, hnum, hnum))
    ret6 = np.zeros((pnum, hnum, hnum, hnum))
    doubleT1 = contract('ci, dj ->cdij', t1, t1)
    for vals in v_ppph:
        c, d, a, k, v = vals
        ret1[a, d] -= v * t1[c, k]
        ret3[d, a] += v * t1[c, k]
        ret0[a, :] -= 0.5 * v * t2[c, d, k, :]
        ret2[c, d, :, k] += v * t1[a, :]
        ret4[a, d, :, k] += v * t1[c, :]
        ret5[a, :, :, k] += v * t2[c, d, :, :]
        ret6[a, :, :, k] += v * doubleT1[c, d, :, :]
    return ret0, ret1, ret2, ret3, ret4, ret5, ret6

#All of the diagrams that appear in t1
def dgram_akci_ck(v_phph, t1):
    """
    Calculates -V^{ak}_{ci}t^{c}_{k}

    :param v_phph:  two body interaction matrix V^{ai}_{bj}
    :type v_phph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('akci, ck -> ai', v_phph, t1)

def dgram_ck_acik(f_ph, t2):
    """
    Calculates f^{c}_{k}t^{ac}_{ik}

    :param f_ph:  Fock matrix f^{a}_{i}
    :type f_ph:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return contract('ck, acik -> ai', f_ph, t2)

def dgram_cikl_cakl(v_phhh, t2):
    """
    Calculates -0.5 * V^{ci}_{kl}t^{ca}_{kl}

    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('cikl, cakl -> ai', v_phhh, t2)

def dgram_cdkl_ck_dali(v_pphh, t1, t2):
    """
    Calculates V^{cd}_{kl}t^{c}_{k}t^{da}_{li}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return contract('cdkl, ck, dali -> ai', v_pphh, t1, t2, optimize = 'greedy')

def dgram_ck_ci(f_ph, t1):
    """
    Calculates -0.5 * f^{c}_{k}t^{c}_{i}

    :param f_ph:  Fock matrix f^{a}_{i}
    :type f_ph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('ck, ci -> ki', f_ph, t1)

def dgram_ck_ak(f_ph, t1):
    """
    Calculates -0.5 * f^{c}_{k}t^{a}_{k}

    :param f_ph:  Fock matrix f^{a}_{i}
    :type f_ph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('ck, ak -> ac', f_ph, t1)

def dgram_bijk_bj(v_phhh, t1):
    """
    Calculates -V^{bi}_{jk}t^{b}_{j}

    :param v_phhh:  two body interaction matrix V^{bi}_{jk}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('bijk, bj -> ki', v_phhh, t1)

def dgram_cdlk_cdli(v_pphh, t2):
    """
    Calculates -0.5 * V^{cd}_{lk}t^{cd}_{li}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('cdlk, cdli -> ki', v_pphh, t2)

def dgram_dckl_dakl(v_pphh, t2):
    """
    Calculates -0.5 * V^{dc}_{kl}t^{da}_{kl}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('dckl, dakl -> ac', v_pphh, t2)

def dgram_cdlk_cl_di(v_pphh, t1):
    """
    Calculates -0.5 * V^{cd}_{lk}t^{c}_{l}t^{d}_{i}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('cdlk, cl, di -> ki',v_pphh, t1, t1, optimize = 'greedy')

def dgram_cdkl_dk_al(v_pphh, t1):
    """
    Calculates 0.5 * V^{cd}_{kl}t^{d}_{k}t^{a}_{l}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return + 0.5 * contract('cdkl, dk, al -> ac',v_pphh, t1, t1, optimize = 'greedy')

def pAB(val):
    """
    Permutator for ab

    :param val:     array to be permutated over
    :type val:      numpy array
    :return:        val^{ab}_{ij} - val^{ba}_{ij}
    :rtype:         numpy array
    """
    return val - contract('abij -> baij', val)

def pIJ(val):
    """
    Permutator for ij

    :param val:     array to be permutated over
    :type val:      numpy array
    :return:        val^{ab}_{ij} - val^{ab}_{ji}
    :rtype:         numpy array
    """
    return val - contract('abij -> abji', val)

def v_pppp_dgrams(v_pppp, t1, t2):
    """
    Calculates both of the diagrams that use v_pppp using the fact that it is sparse

    :param v_pppp:  nonzeroes of the two body interaction matrix V^{ab}_{cd}
    :type v_pppp:   list[(int, int, int, int, float)]
    :param t1:      T1 matrix t_i^a
    :type t1:       numpy array
    :param t2:      T2 matrix t_{ij}^{ab}
    :type t2:       numpy array
    :return:        The result of the two diagrams that contribute to t2
    :rtype:         numpy array, numpy array
    """
    hnum = np.shape(t1)[1]
    pnum = np.shape(t1)[0]
    ret1 = np.zeros((pnum, pnum,hnum, hnum))
    ret2 = np.zeros((pnum, pnum,hnum, hnum))
    doubleT1 = contract('ci, dj ->cdij', t1, t1)
    for val in v_pppp:
        a, b, c, d, v = val
        ret1[a, b, :, :] += v * t2[c, d, :, :]
        ret2[a, b, :, :] += v * doubleT1[c, d, : , :]
    return ret1, ret2

#All of the diagrams that appear in t2
def dgram_klij_abkl(v_hhhh, t2):
    """
    Calculates 0.5 * V^{kl}_{ij}t^{ab}_{kl}

    :param v_hhhh:  two body interaction matrix V^{kl}_{ij}
    :type v_hhhh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * contract('klij, abkl -> abij', v_hhhh, t2)

def dgram_bkcj_acik(v_phph, t2):
    """
    Calculates -P(ij)P(ab)V^{bk}_{cj}t^{cb}_{ik}

    :param v_phph:  two body interaction matrix V^{bk}_{cj}
    :type v_phph:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - pIJ(pAB(contract('bkcj, acik -> abij', v_phph, t2)))

def dgram_bkij_ak(v_phhh, t1):
    """
    Calculates P(ab)V^{bk}_{ij}t^{a}_{k}

    :param v_phhh:  two body interaction matrix V^{bk}_{ij}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return pAB(contract('bkij, ak -> abij', v_phhh, t1))

def dgram_cdkl_acik_dblj(v_pphh, t2):
    """
    Calculates 0.5 * P(ij)P(ab)V^{cd}_{kl}t^{ac}_{ik}t^{db}_{lj}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pIJ(pAB(contract('cdkl, acik, dblj -> abij', v_pphh, t2, t2, optimize="greedy")))

def dgram_cdkl_cdij_abkl(v_pphh, t2):
    """
    Calculates 0.25 * P(ab)V^{cd}_{kl}t^{cd}_{ij}t^{ab}_{kl}

    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.25 * contract('cdkl, cdij, abkl -> abij', v_pphh, t2, t2, optimize="greedy")

def dgram_klij_ak_bl(v_hhhh, t1):
    """
    Calculates - 0.5 * P(ab)V^{kl}_{ij}t^{a}_{k}t^{b}_{l}

    :param v_hhhh:  two body interaction matrix V^{kl}_{ij}
    :type v_hhhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pAB(contract('klij, ak, bl -> abij', v_hhhh, t1, t1, optimize="greedy"))

def dgram_bkci_ak_cj(v_phph, t1):
    """
    Calculates - P(ij)P(ab)V^{bk}_{ci}t^{a}_{k}t^{c}_{j}

    :param v_phph:  two body interaction matrix V^{ai}_{bj}
    :type v_phph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - pAB(pIJ(contract('bkci, ak, cj -> abij', v_phph, t1, t1, optimize="greedy")))

def dgram_cikl_ck_ablj(v_phhh, t1, t2):
    """
    Calculates - P(ij)V^{ci}_{kl}t^{c}_{k}t^{ab}_{lj}

    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - pIJ(contract('cikl, ck, ablj -> abij', v_phhh, t1, t2, optimize="greedy"))

def dgram_da_dbij(v_ppph_res, t2):
    """
    Calculates - P(ab)X^d_at^{db}_{ij}

    :param v_ppph_res:  intermediate result 3 from :meth:`v_ppph_dgrams`
    :type v_ppph_res:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - pAB(contract('da, dbij -> abij', v_ppph_res, t2))

def dgram_acik_bcjk(v_ppph_res, t2):
    """
    Calculates - P(ij)P(ab)X^{ac}_{ij}t^{bc}_{jk}

    :param v_ppph_res:  intermediate result 4 from :meth:`v_ppph_dgrams`
    :type v_ppph_res:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return pIJ(pAB(contract('acik, bcjk -> abij', v_ppph_res, t2)))

def dgram_cikl_al_bcjk(v_phhh, t1, t2):
    """
    Calculates - P(ij)P(ab)V^{ci}_{kl}t^{a}_{l}t^{bc}_{jk}

    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - pIJ(pAB(contract('cikl, al, bcjk -> abij', v_phhh, t1, t2, optimize="greedy")))

def dgram_cjkl_ci_abkl(v_phhh, t1, t2):
    """
    Calculates 0.5 * P(ij)V^{cj}_{kl}t^{c}_{i}t^{ab}_{kl}

    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pIJ(contract('cjkl, ci, abkl -> abij', v_phhh, t1, t2, optimize="greedy"))

#note that these two functions have the same indices, but are 
#from two different diagrams, and that is because I factor out 
#parts of them when calculating the v_ppph diagrams
def dgram_bijk_ak1(v_ppph_res, t1):
    """
    Calculates 0.5 * P(ab)X^{bi}_{jk}t^{a}_{k}

    :param v_ppph_res:  intermediate result 5 from :meth:`v_ppph_dgrams`
    :type v_ppph_res:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pAB(contract('bijk, ak -> abij', v_ppph_res, t1))

def dgram_bijk_ak2(v_ppph_res, t1):
    """
    Calculates 0.5 * P(ij)P(ab)X^{bi}_{jk}t^{a}_{k}

    :param v_ppph_res:  intermediate result 6 from :meth:`v_ppph_dgrams`
    :type v_ppph_res:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pIJ(pAB(contract('bijk, ak -> abij', v_ppph_res, t1)))

def dgram_cjkl_ci_ak_bl(v_phhh, t1):
    """
    Calculates 0.5 * P(ij)P(ab)V^{cj}_{kl}t^{c}_{i}t^{a}_{k}t^{b}_{l}

    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.5 * pIJ(pAB(contract('cjkl, ci, ak, bl -> abij', v_phhh, t1, t1, t1, optimize="greedy")))

def dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2):
    """
    Calculates 0.25 * P(ij)V^{cd}_{kl}t^{c}_{i}t^{d}_{j}t^{ab}_{kl}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.25 * pIJ(contract('cdkl, ci, dj, abkl -> abij', v_pphh, t1, t1 ,t2, optimize="greedy"))

def dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2):
    """
    Calculates 0.25 * P(ab)V^{cd}_{kl}t^{a}_{k}t^{b}_{l}t^{cd}_{ij}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.25 * pAB(contract('cdkl, ak, bl, cdij -> abij', v_pphh, t1, t1 ,t2, optimize="greedy"))

def dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2):
    """
    Calculates P(ij)P(ab)V^{cd}_{kl}t^{c}_{i}t^{b}_{l}t^{ad}_{kj}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return pIJ(pAB(contract('cdkl, ci, bl, adkj -> abij', v_pphh, t1, t1, t2, optimize="greedy")))

def dgram_cdkl_ci_ak_dj_bl(v_pphh, t1):
    """
    Calculates 0.25 * P(ij)P(ab)V^{cd}_{kl}t^{c}_{i}t^{a}_{k}t^{d}_{j}t^{b}_{l}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return 0.25 * pIJ(pAB(contract('cdkl, ci, ak, dj, bl -> abij', v_pphh, t1, t1, t1, t1, optimize="greedy")))

def dgram_cdkl_bdkl(v_pphh, t2):
    """
    Calculates -0.5 * V^{cd}_{kl}t^{bd}_{kl}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('cdkl, bdkl -> bc', v_pphh, t2)

def dgram_cdkl_cdjl(v_pphh, t2):
    """
    Calculates -0.5 * V^{cd}_{kl}t^{cd}_{jl}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - 0.5 * contract('cdkl, cdjl -> kj', v_pphh, t2)

def dgram_ck_bk(f_ph, t1):
    """
    Calculates - * f^{c}_{k}t^{b}_{k}

    :param f_ph:  Fock matrix f^{a}_{i}
    :type f_ph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('ck, bk -> bc', f_ph, t1)

def dgram_ck_cj(f_ph, t1):
    """
    Calculates - * f^{c}_{k}t^{c}_{j}

    :param f_ph:  Fock matrix f^{a}_{i}
    :type f_ph:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('ck, cj -> kj', f_ph, t1)

def dgram_cdlk_cl_dj(v_pphh, t1):
    """
    Calculates -V^{cd}_{kl}t^{c}_{l}t^{d}_{j}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('cdlk, cl, dj -> kj', v_pphh, t1, t1, optimize="greedy")

def dgram_cdlk_dk_bl(v_pphh, t1):
    """
    Calculates -V^{cd}_{kl}t^{d}_{k}t^{b}_{l}

    :param v_pphh:  two body interaction matrix V^{ab}_{jk}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :return:        result of the contraction
    :rtype:         numpy array
    """
    return - contract('cdlk, dk, bl -> bc', v_pphh, t1, t1, optimize="greedy")