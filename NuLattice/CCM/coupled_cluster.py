"""
Provides functions for setting up the coupled cluster equations on the lattice and solving for them
"""
__authors__   =  ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__credits__   =  ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__copyright__ = "(c) Thomas Papenbrock and Maxwell Rothman and Ben Johnson"
__license__   = "BSD-3-Clause"
__date__      = "2025-09-03"


import numpy as np
from opt_einsum import contract
from copy import deepcopy
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".." / ".."))
import NuLattice.CCM.three_body_utils as tbu
import NuLattice.lattice as lat
import NuLattice.CCM.ccDgrams as dgrams

def get_fock_matrices(part,hole,myTkin,v_phph,v_phhh,v_hhhh):
    """
    constructs fock matrices from one-body interaction and rank-4 tensors of two-body force

    :param part:    list of particle indices
    :type part:     list[int]
    :param hole:    list of hole indices
    :type hole:     list[int]
    :param myTkin:  list of one-body matrix elements
    :type myTkin:   list[(int, int, float)]
    :param v_phph:  two body interaction matrix V^{ai}_{bj}
    :type v_phph:   numpy array
    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param v_hhhh:  two body interaction matrix V^{ij}_{kl}
    :type v_hhhh:   numpy array
    :return:    Fock matrices f_pp, f_ph, f_hh
    :rtype:     numpy array, numpy array, numpy array
    """
    # Load kinetic energy into dictionary for fast lookup
    lookup1b = {}
    for ele in myTkin:
        [p, q, val] = ele
        if (p, q) in lookup1b.keys():
            lookup1b[(p, q)] += val
        else:
            lookup1b[(p,q)]= val
    pnum=len(part)
    hnum=len(hole)
    
    f_pp = np.zeros((pnum,pnum))
    f_ph = np.zeros((pnum,hnum))
    f_hh = np.zeros((hnum,hnum))

    # Build particle-particle fock matrix
    for a in range(pnum):
        ka=part[a]
        for b in range(pnum):
            kb=part[b]
            labels=(ka,kb)
            val = lookup1b.get( labels )
            if val == None:
                continue
            else:
                f_pp[a,b] = val

            f_pp[a,b] = f_pp[a,b] + np.sum( [ v_phph[a,i,b,i] for i in range(hnum) ] )
                   
    # Build particle-hole fock matrix
    for a in range(pnum):
        ka=part[a]
        for b in range(hnum):
            kb=hole[b]
            labels=(ka,kb)
            val = lookup1b.get( labels )
            if val == None:
                continue
            else:
                f_ph[a,b] = val

            f_ph[a,b] = f_ph[a,b] + np.sum( [ v_phhh[a,i,b,i] for i in range(hnum) ] )       
        
    # Build hole-hole fock matrix
    for a in range(hnum):
        ka=hole[a]
        for b in range(hnum):
            kb=hole[b]
            labels=(ka,kb)
            val = lookup1b.get( labels )
            if val == None:
                continue
            else:
                f_hh[a,b] = val

            f_hh[a,b] = f_hh[a,b] + np.sum( [ v_hhhh[a,i,b,i] for i in range(hnum) ] )
            
    return f_pp, f_ph, f_hh

def get_all_interactions(part,hole,mycontact, sparse = False):
    """
    This routine takes the relatively small number of two-body matrix elements in mycontact
    and sorts them into the four-indexed interaction tensors. It also anti-symmetrizes the latter
    whenin and out indices run over the same set of particle/hole indices.

    :param part:    list of particle-space indices
    :type part:     list[int]
    :param hole:    list of hole-space indices
    :type hole:     list[int]
    :param mycontact:   list of two-body matrix elements
    :type mycontact:    list[(int, int, int, int, float)]
    :param sparse:      Optional; whether or not v_pppp and v_ppph should be stored as sparse arrays or not
    :type sparse:       bool
    :return:    Two body matrices v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh as rank-four tensors.  
    :rtype:     numpy array | sparse array, numpy array | sparse array, numpy array, numpy array, numpy array, numpy array
    """
    pnum=len(part)
    hnum=len(hole)
    
    # Lookup from general index to hole index
    vals     = range(hnum)
    lookup_h = dict(zip(hole,vals))

    # Lookup from general index to particle index
    vals     = range(pnum)
    lookup_p = dict(zip(part,vals))

    # Handle pppp and ppph blocks differently if we are doing a sparse calculation
    if sparse:
        v_pppp = []
        v_ppph = []
    else:
        v_pppp = np.zeros((pnum,pnum,pnum,pnum))
        v_ppph = np.zeros((pnum,pnum,pnum,hnum))

    v_pphh=np.zeros((pnum,pnum,hnum,hnum))
    v_phph=np.zeros((pnum,hnum,pnum,hnum))
    v_phhh=np.zeros((pnum,hnum,hnum,hnum))
    v_hhhh=np.zeros((hnum,hnum,hnum,hnum))

    # Loop to populate v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh
    # from mycontact
    #
    # We execute the following steps:
    # 1) identify which block we are in
    # 2) permute hole/particles in bra/ket to get to the right block above
    # 3) we then assemble all antisymmetric permutations
    # 4) we finally write the matrix element to the appropriate block, either dense or sparse
    #
    # This approach works with the p<q, r<s storage of mycontact
    # and prevents double counting matrix elements in the final tensors
    for [i1,i2,i3,i4, val] in mycontact:
        currSparse = False
        # note: i1<i2 and i3<i4 is stored only in mycontact
        ket=[]
        bra=[]
        if i1 in hole:
            ket.append("h")
        else:
            ket.append("p")
      
        if i2 in hole:
            ket.append("h")
        else:
            ket.append("p")
        
        if i3 in hole:
            bra.append("h")
        else:
            bra.append("p")
      
        if i4 in hole:
            bra.append("h")
        else:
            bra.append("p")
        
        ket = tuple(ket)
        bra = tuple(bra)
    
        if ket == ("p","p"):
            sign_ket = 1.0
            a = lookup_p.get(i1)
            b = lookup_p.get(i2)
        elif ket == ("p","h"):
            sign_ket = 1.0
            a = lookup_p.get(i1)
            b = lookup_h.get(i2)
        elif ket == ("h","h"):
            sign_ket = 1.0
            a = lookup_h.get(i1)
            b = lookup_h.get(i2)
        elif ket == ("h","p"):
            sign_ket = -1.0
            b = lookup_h.get(i1)
            a = lookup_p.get(i2)
            ket = ("p","h")
        
        ket_indx=(a,b)
    
        if bra == ("p","p"):
            sign_bra = 1.0
            c = lookup_p.get(i3)
            d = lookup_p.get(i4)
        elif bra == ("p","h"):
            sign_bra = 1.0
            c = lookup_p.get(i3)
            d = lookup_h.get(i4)
        elif bra == ("h","h"):
            sign_bra = 1.0
            c = lookup_h.get(i3)
            d = lookup_h.get(i4)
        elif bra == ("h","p"):
            sign_bra = -1.0
            d = lookup_h.get(i3)
            c = lookup_p.get(i4)
            bra = ("p","h")
    
        bra_indx=(c,d)
    
    
        # in what follows, vint is used as a pointer, i.e. as a view of a numpy array
        if ket == ("p","p"):
            if bra == ("p","p"): 
                vint = v_pppp
                indices= ((a,b,c,d),(b,a,c,d),(a,b,d,c),(b,a,d,c))
                signs  = (1.0,-1.0,-1.0,1.0)
                currSparse = sparse
            elif bra == ("p","h"): 
                vint = v_ppph
                indices= ((a,b,c,d),(b,a,c,d))
                signs  = (1.0,-1.0)
                currSparse = sparse
            elif bra == ("h","h"): 
                vint = v_pphh
                indices= ((a,b,c,d),(b,a,c,d),(a,b,d,c),(b,a,d,c))
                signs  = (1.0,-1.0,-1.0,1.0)
        elif ket == ("p","h"):
            if bra == ("p","p"):
                vint = None
            else:
                if bra == ("p","h"): 
                    vint = v_phph
                    indices=[(a,b,c,d)]
                    signs  = [1.0]
                elif bra == ("h","h"): 
                    vint = v_phhh
                    indices= ((a,b,c,d),(a,b,d,c))
                    signs  = (1.0,-1.0)
        elif ket == ("h","h"):
            if bra == ("h","h"):
                indices= ((a,b,c,d),(b,a,c,d),(a,b,d,c),(b,a,d,c))
                signs  = (1.0,-1.0,-1.0,1.0)
                vint = v_hhhh
            else:
                vint = None

        if vint is not None:
            for i, indx in enumerate(indices):
                sign = signs[i]
                if currSparse:
                    vint.append([indx[0], indx[1], indx[2], indx[3], val * sign_ket * sign_bra * sign])
                else:
                    vint[indx] =  val * sign_ket * sign_bra * sign
    

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh

def ccsd_energy(f_ph, v_pphh, t2, t1):
    """
    computes ccsd correlation energy
    Note: Technically, this would need v_hhpp but this is of course the transpose of v_pphh; 
    Likewise, f_hp is required, and this is just the transpose of f_ph which we have

    :param f_ph:    Fock matrix f^{a}_{i}
    :type f_ph:     numpy array
    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param t1:      T^{a}_{i} from the coupled cluster equations
    :type t1:       numpy array
    :param t2:      T^{ab}_{ij} from the coupled cluster equations
    :type t2:       numpy array
    :return:        CCSD correlation energy
    :rtype:         float
    """
    erg = ( contract( 'ai,ai', f_ph, t1 )
           + contract( 'abij,abij', v_pphh, t2 )*0.25
           + contract( 'abij,ai,bj', v_pphh, t1, t1, optimize='greedy' )*0.5
            )
    return erg

def get_ref_energy(no_1b_hh, no_2b_hhhh, w_hhh_hhh=None):
    """
    Computes the energy of the reference state from normal ordered interactions

    :param no_1b_hh:    Fock matrix (including contributions from 3NF if w_hhh_hhh is not None)
    :type no_1b_hh:     numpy array
    :param no_2b_hhhh:  numpy array, normal-ordered TBME including contributions from 3NF if w_3b is not None
    :type no_2b_hhhh:   numpy array
    :param w_hhh_hhh:   list of nonzero elements of 3NF or None if not using
    :type w_hhh_hhh:    list[(int, int, int, int, int, int, float)]
    :return: energy of reference state
    :rtype: float

    """
    en = 0.0
    hnum = len(no_1b_hh)
    # Because no_1b_hh and no_2b_hh are already normal ordered,
    # we use modified expressions to compute the reference state energy
    # [see, e.g., Frosini et al. EPJA 57 (2021)]
    for i in range(hnum):
        en+= no_1b_hh[(i,i)]
        for j in range(hnum):
            en -= 0.5 * no_2b_hhhh[(i,j,i,j)]

    if w_hhh_hhh != None:
        for ele in w_hhh_hhh:
            [m, i, j, n, k, l, val] = ele
            if (m, i, j) == (n, k, l):
                en  += val / 6.0
    return en   

def t1Init(f_ph, f_pp, f_hh, delta):
    """
    Initializes t_i^a based on the perturbation theory guess

    :param f_ph:    Fock matrix f^a_i
    :type f_ph:     numpy array
    :param f_pp:    Fock matrix f^a_b
    :type f_pp:     numpy array
    :param f_hh:    Fock matrix f^i_j
    :type f_hh:     numpy array
    :param delta:   Energy gap to make sure there is no division by 0 errors
    :type delta:    float
    :return:        t_i^a initial guess
    :rtype:         numpy array
    """
    diag_h = np.diag(f_hh)
    diag_p =  - np.diag(f_pp)
    # Build energy differences diff_ph = -1 * e_p + e_h + delta
    # where the delta (usually 0) can be used to protect against
    # nearly degenerate hole vs. particle energies
    denom  = np.add.outer(diag_p, diag_h) + delta
    return f_ph / denom

def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, 
           v_ppph_results, sparse = True):
    """
    iterating t1 using the CCSD equations

    :param t1:      t^a_i from the previous iteration
    :type t1:       numpy array
    :param t2:      t^{ab}_{ij} from the previous iteration
    :type t2:       numpy array
    :param f_ph:    Fock matrix f^a_i
    :type f_ph:     numpy array
    :param f_pp:    Fock matrix f^a_b
    :type f_pp:     numpy array
    :param f_hh:    Fock matrix f^i_j
    :type f_hh:     numpy array
    :param v_phph:  two body interaction matrix V^{ai}_{bj}
    :type v_phph:   numpy array
    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param v_ppph_results:  two body interaction matrix V^{ab}_{ci} if sparse = False, 
                            results from :meth:`CCM.ccDgrams.v_ppph_dgrams` otherwise
    :type v_ppph_results:   numpy array | list[numpy array]
    :param sparse:  whether or not v_pppp and v_ppph are stored as sparse arrays or not
    :type sparse:   bool    
    :return:     updated t_i^a
    :rtype:     numpy array
    """
    #Calculating H_i^a without the X_i^i and X_a^a terms
    H1 = np.zeros_like(f_ph)
    H1 += (f_ph 
             + dgrams.dgram_akci_ck(v_phph, t1)
             + dgrams.dgram_ck_acik(f_ph, t2)
             + dgrams.dgram_cikl_cakl(v_phhh, t2)
             + dgrams.dgram_cdkl_ck_dali(v_pphh, t1, t2)
            )
    

    # #Calculating X_hh and X_pp, where they are the factorization 
    # #note, I really calculate -X_i^i here
    pnum = len(f_pp)
    hnum = len(f_hh)

    X_hh = np.zeros((hnum, hnum))
    X_pp = np.zeros((pnum, pnum))
    
    X_hh -= f_hh #2
    X_pp += f_pp #3
    
    # #if factoring, add all of the relavent terms to X_i^i and X_a^a
    # We "dress" the Fock matrix with additional contributions from CC diagrams
    X_hh += dgrams.dgram_ck_ci(f_ph, t1) #8
    X_pp += dgrams.dgram_ck_ak(f_ph, t1) #8
    X_hh += dgrams.dgram_bijk_bj(v_phhh, t1) #9
    X_hh += dgrams.dgram_cdlk_cdli(v_pphh, t2) 
    X_pp += dgrams.dgram_dckl_dakl(v_pphh, t2)
    X_hh += dgrams.dgram_cdlk_cl_di(v_pphh, t1)
    X_pp += dgrams.dgram_cdkl_dk_al(v_pphh, t1)
    
    if sparse:
        H1 += v_ppph_results[0]
        X_pp += v_ppph_results[1]
    else:
        H1 += - 0.5 * contract('cdak, cdki -> ai', v_ppph_results, t2)#6
        X_pp -= contract('cdak, ck -> ad', v_ppph_results, t1) #10
    
    H1  += contract('ac, ci -> ai', X_pp, t1) + contract('ki, ak -> ai', X_hh, t1)

    #divides H_i^a by X_i^i - X_a^a to iterate t1
    diag_h = np.diag(X_hh)
    diag_p = np.diag(X_pp)
    denom = - np.add.outer(diag_p, diag_h)
    t1 += H1 / denom
    return t1

def t2Init(f_pp, f_hh, v_pphh, delta):
    """
    Initializes t_{ij}^{ab} based on perturbation theory guess
    
    :param f_pp:    Fock matrix
    :type f_pp:     numpy array
    :param f_hh:    Fock matrix
    :type f_hh:     numpy array
    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param delta:   Energy gap to avoid division by 0 errors
    :type delta:    float
    :return:        t^{ab}_{ij} based on perturbation theory guess
    :rtype:         numpy array
    """
    diag_h = np.diag(f_hh)
    diag_p = np.diag(f_pp)
    denom_hh = np.add.outer(diag_h, diag_h)
    denom_pp = - np.add.outer(diag_p, diag_p)
    # Build energy differences diff_pphh = -1 * (e_p + e_p') + e_h + e_h' + delta
    # where the delta (usually 0) can be used to protect against
    # nearly degenerate hole vs. particle energies
    denom = np.add.outer(denom_pp, denom_hh) + delta    
    return v_pphh / denom


def t2Iter(t1, t2, f_ph, f_hh, f_pp, v_pppp, v_phph, v_phhh, v_pphh, 
           v_ppph_results, v_hhhh, sparse = True):
    """
    iterating t2, factoring out terms that look like g_i^i and g_a^a

    :param t1:  t^a_i from the previous iteration
    :type t1:   numpy array
    :param t2:  t^{ab}_{ij} from the previous iteration
    :type t2:   numpy array
    :param f_ph:    Fock matrix f_i^a
    :type f_ph:     numpy array
    :param f_hh:    Fock matrix f_i^j
    :type f_hh:     numpy array
    :param f_pp:    Fock matrix f_a^b
    :type f_pp:     numpy array
    :param v_pppp:  two body interaction matrix V^{ab}_{cd}; optionally stored as a sparse array
    :type v_pppp:   numpy array | list[(int, int, int, int, float)]
    :param v_phph:  two body interaction matrix V^{ai}_{bj}
    :type v_phph:   numpy array
    :param v_phhh:  two body interaction matrix V^{ai}_{jk}
    :type v_phhh:   numpy array
    :param v_pphh:  two body interaction matrix V^{ab}_{ij}
    :type v_pphh:   numpy array
    :param v_ppph_results:  if not sparse, the two body interaction matrix V^{ab}_{ci}, but if 
                            sparse=True, it is the contributions to T2 from the V^{ab}_{ci} diagrams
    :type v_ppph_results:   numpy array | list[numpy array]
    :param v_hhhh:  two body interaction matrix V^{ij}_{kl}
    :type v_hhhh:   numpy array
    :param sparse:  whether or not v_pppp and v_ppph are stored as sparse arrays or not
    :type sparse:   bool    
    :return:     updated t_ij^ab
    :rtype:     numpy array
    """
    #Adding together all of the terms in H_ij^ab that doesn't get factored out
    H2 = np.zeros_like(v_pphh)
    H2 += v_pphh
    H2 += dgrams.dgram_klij_abkl(v_hhhh, t2)
    H2 += dgrams.dgram_bkcj_acik(v_phph, t2) 
            
    H2 += dgrams.dgram_bkij_ak(v_phhh, t1) 
            
    H2 += dgrams.dgram_cdkl_acik_dblj(v_pphh, t2)
    H2 += dgrams.dgram_cdkl_cdij_abkl(v_pphh, t2)
    H2 += dgrams.dgram_klij_ak_bl(v_hhhh, t1) 
    H2 += dgrams.dgram_bkci_ak_cj(v_phph, t1) 
    H2 += dgrams.dgram_cikl_ck_ablj(v_phhh, t1, t2) 
            
    H2 += dgrams.dgram_cikl_al_bcjk(v_phhh, t1, t2) 
    H2 += dgrams.dgram_cjkl_ci_abkl(v_phhh, t1, t2) 
            
    H2 += dgrams.dgram_cjkl_ci_ak_bl(v_phhh, t1) 
    H2 += dgrams.dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2) 
    H2 += dgrams.dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2) 
    H2 += dgrams.dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2) 
    H2 += dgrams.dgram_cdkl_ci_ak_dj_bl(v_pphh, t1)
    
    pnum = len(f_pp)
    hnum = len(f_hh)

    X_hh = np.zeros((hnum, hnum))
    X_pp = np.zeros((pnum, pnum))
    #factoring out all of the X_a^a and X_i^i terms
    #note that I am again calculating -X_i^i here like for t1
    X_hh -= f_hh
    X_pp += f_pp

    #adding up everything that goes in the factored terms
    # We "dress" the Fock matrix with additional contributions from CC diagrams
    X_pp += dgrams.dgram_cdkl_bdkl(v_pphh, t2)
    X_hh += dgrams.dgram_cdkl_cdjl(v_pphh, t2)
    X_pp += dgrams.dgram_ck_bk(f_ph, t1) 
    X_hh += dgrams.dgram_ck_cj(f_ph, t1) 
    X_hh += dgrams.dgram_cdlk_cl_dj(v_pphh, t1) 
    X_pp += dgrams.dgram_cdlk_dk_bl(v_pphh, t1) 

    if sparse:
        H2 += dgrams.pIJ(v_ppph_results[2])
        H2 += dgrams.dgram_da_dbij(v_ppph_results[3], t2)
        H2 += dgrams.dgram_acik_bcjk(v_ppph_results[4], t2)
        H2 += dgrams.dgram_bijk_ak1(v_ppph_results[5], t1)
        H2 += dgrams.dgram_bijk_ak2(v_ppph_results[6], t1)

        dgram_abcd_cdij, dgram_abcd_ci_dj = dgrams.v_pppp_dgrams(v_pppp, t1, t2)
        H2 += 0.5 * dgram_abcd_cdij
        H2 += 0.5 * dgrams.pIJ(dgram_abcd_ci_dj)
    else:
        H2 += dgrams.pIJ(contract('abcj, ci -> abij', v_ppph_results, t1))
        H2 += - dgrams.pAB(contract('cdak, ck, dbij -> abij', v_ppph_results, t1, t2, optimize="greedy"))
        H2 += dgrams.pIJ(dgrams.pAB(contract('dcak, di, bcjk -> abij', v_ppph_results, t1, t2, optimize="greedy")))
        H2 += 0.5 * dgrams.pAB(contract('cdbk, ak, cdij -> abij', v_ppph_results, t1, t2, optimize="greedy"))
        H2 += 0.5 * dgrams.pIJ(dgrams.pAB(contract('cdbk, ci, ak, dj -> abij', v_ppph_results, t1, t1, t1, optimize="greedy")))
        H2 += 0.5 * contract('abcd, cdij -> abij', v_pppp, t2)
        H2 += 0.5 * dgrams.pIJ(contract('abcd, ci, dj -> abij', v_pppp, t1, t1, optimize="greedy"))

    H2 += dgrams.pAB(contract('bc, acij -> abij', X_pp, t2)) + dgrams.pIJ(contract('kj, abik -> abij', X_hh, t2))
    
    #fast calculation of the denominator X_ii+X_jj-X_aa-X_bb
    diag_h = np.diag(X_hh)
    diag_p = np.diag(X_pp)
    denom_hh = np.add.outer(diag_h, diag_h)
    denom_pp = np.add.outer(diag_p, diag_p)
    denom = - np.add.outer(denom_pp, denom_hh)
    t2 += H2 / denom
    
    return t2

def ccsd_solver(fock_mats, two_body_int, t1initial=None, eps = 1e-8, maxSteps = 1000, max_diis = 10, 
                delta = 0, mixing = 0.5, verbose = False, sparse = True, ccs = False): 
    """
    Solves for the correlation energy of a system using the CCSD equations.
    DIIS credit to Daniel G. A. Smith and Lori A. Burns, and The Psi4NumPy Developers, https://github.com/psi4/psi4numpy

    :param fock_mats:   Fock matrices
    :type fock_mats:    list[numpy array]
    :param two_body_int:    two body interaction matrices; if sparse the first two elements will be lists instead of numpy arrays
    :type two_body_int: list[numpy array]
    :param eps:     Optional; max relative error
    :type eps:      float
    :param maxSteps:    Optional; max number of iterations to take
    :type maxSteps:     int
    :param max_diis:    Optional; size of DIIS array. Set to 0 to not perform DIIS
    :type max_diis:     int
    :param delta:   Optional; energy gap for first iteration to avoid division by 0 errors
    :type delta:    float
    :param mixing:  Optional; how much of the previous step to mix into the current one. Set to 0 if you
                    don't want any mixing from the previous step
    :type mixing:   float between 0 and 1
    :param verbose: Optional; whether or not to print steps as they are calculated
    :type verbose:  bool
    :param sparse:  Optional; whether or not v_pppp and v_ppph are stored as sparse arrays or not
    :type sparse:   bool
    :param ccs:     Optional; whether to perform just the ccs equations or not
    :type ccs:      bool
    :returns:   correlation energy, t1, and t2
    :rtype:     float, numpy array, numpy array
    """
    f_pp, f_ph, f_hh = fock_mats
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int

    # Initialize T1 and T2 to perturbative guess if not yet initialized
    if t1initial is None:
        t1 = t1Init(f_ph, f_pp, f_hh, delta)
    else:
        t1 = t1initial
    if ccs or t1initial is not None:
        t2 = np.zeros_like(v_pphh)
    else:
        t2 = t2Init(f_pp, f_hh, v_pphh, delta)

    if max_diis > 0:
        diis_vals_t1 = [deepcopy(t1)]
        diis_vals_t2 = [deepcopy(t2)]
        diis_errors = []
        
    prevEnergy = ccsd_energy(f_ph,v_pphh, t2, t1)
    
    if verbose:
        print(f'Step {0}: {prevEnergy}')

    #iterate
    for i in range(maxSteps):
        oldT1 = deepcopy(t1)
        oldT2 = deepcopy(t2)
                
        # If we are doing sparse calculations, we have factorized out the diagrams 
        # that contract v_ppph with t1 and t2 to intermediates.
        # Otherwise, we let opt_einsum do the factorization automatically.
        if sparse:
            v_ppph_results = dgrams.v_ppph_dgrams(v_ppph, t1, t2)
        else:
            v_ppph_results = v_ppph

        # Iterate T1 and T2 
        t1 = mixing * t1 + (1. - mixing) * t1Iter(t1, t2, f_ph, f_pp, f_hh,
                                                  v_phph, v_phhh, v_pphh, v_ppph_results,
                                                  sparse=sparse)
        if not ccs:
            t2 = mixing * t2 + (1. - mixing) * t2Iter(oldT1, t2, f_ph, f_hh, f_pp,
                                                      v_pppp, v_phph, v_phhh, v_pphh, v_ppph_results, v_hhhh,
                                                      sparse=sparse)

        # Compute new CCSD energy
        energy = ccsd_energy(f_ph,v_pphh, t2, t1)

        if verbose:
            print(f'Step {i+1}: {energy}', "difference =", abs(energy - prevEnergy) / abs(energy))

        # Check if we are done
        if (energy == 0 and prevEnergy == 0) or abs(energy - prevEnergy) / abs(energy) < eps:
            if verbose:
                print(f'Energy found in {i+1} iterations')
            return energy, t1, t2
        
        # Handle convergence acceleration via DIIS
        diis_vals_t1.append(deepcopy(t1))
        diis_vals_t2.append(deepcopy(t2))

        error_t1 = (t1 - oldT1).ravel()
        error_t2 = (t2 - oldT2).ravel()
        diis_errors.append(np.concatenate((error_t1, error_t2)))
        if max_diis > 0 and len(diis_errors) == max_diis:
            diis_size = len(diis_vals_t1)
            del diis_vals_t1[0]
            del diis_vals_t2[0]
            del diis_errors[0]
                
            diis_size -= 1
            B = -1 * np.ones((diis_size, diis_size))
            B[-1, -1] = 0

            for n1, e1 in enumerate(diis_errors):
                for n2, e2 in enumerate(diis_errors):
                    # Vectordot the error vectors
                    B[n1, n2] = np.dot(e1, e2)

            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            resid = np.zeros(diis_size)
            resid[-1] = -1

            ci = np.linalg.solve(B, resid)

            t1[:] = 0
            if not ccs:
                t2[:] = 0
            for num in range(diis_size - 1):
                t1 += ci[num] * diis_vals_t1[num + 1]
                if not ccs:
                    t2 += ci[num] * diis_vals_t2[num + 1]
            diis_vals_t1 = [deepcopy(t1)]
            diis_vals_t2 = [deepcopy(t2)]
            diis_errors = []
        
        # Catch if the iterations are going off to infinity
        if abs(energy) > 1.e+10:
            print('Iteration Diverged')
            break

        # Catch if there is a divide by 0 error or similar
        if np.isnan(energy):
            print('nan error')
            break
            
        prevEnergy = energy
        if i == maxSteps - 1:
            print('Max Iterations Reached')
            return energy, t1, t2

        
def get_norm_ord_int(thisL, holes, vT1, vS1, str_3NF = 0, sparse = True):
    """
    Takes all the necessary parameters to generate the fock matrices, two and three body interactions to perform CCM

    :param thisL:   number of lattice sites in each direction
    :type thisL:    int
    :param holes:   list of holes in the format [x, y, z, tz+0.5, sz+0.5]. To generate these, see :meth:`lattice.makeState`
    :type holes:    list[list[int]]
    :param vT1:     strength of T=1 coupling
    :type vT1:      float
    :param vS1:     strength of S=1 coupling
    :type vS1:      float
    :param str_3NF: Optional strength of T=1 coupling
    :type str_3NF:  float
    :param sparse:  Optional whether or not to store v_pppp and v_ppph as a sparse array or not
    :type sparse:   bool
    :return:        The reference energy and three lists, a list of the three fock matrices in the order f_pp, f_ph, f_hh, 
                    all of the two body interactions in the order v_pppp, v_ppph, v_pphh, v_phph, 
                    v_phhh, v_hhhh
    """
    #setting up the lattice
    lattice = lat.get_lattice(thisL)
    myTkin=lat.Tkin(lattice, thisL)
    mycontact=lat.contacts(vT1, vS1, lattice, thisL)
    hole, part = lat.states2PHSpace(holes, thisL)

    #Getting the 2 body interactions and fock matrices
    two_body_int = list(get_all_interactions(part,hole,mycontact, sparse))
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int
    fock_mats = list(get_fock_matrices(part, hole, myTkin, v_phph, v_phhh, v_hhhh))
    f_pp, f_ph, f_hh = fock_mats

    three_body_int = None
    #getting the 3 body interactions if needed and adding the effective contribution to the fock matrices and two body interactions
    if str_3NF != 0:
        my3body = lat.NNNcontact(str_3NF, lattice, thisL)
        three_body_int = list(tbu.get_3NF(part, hole, my3body)) #order here is w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, 
                                                                #w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh
        w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh = three_body_int

        pnum, hnum = np.shape(f_ph)
        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_phh_phh, w_phh_hhh, w_hhh_hhh)

        for i in range(len(dum_fock)):
            fock_mats[i] += dum_fock[i]

        dum_two_body = tbu.get_3NF_tbme(w_pph_pph, w_pph_phh, w_pph_hhh, 
                                                w_phh_phh, w_phh_hhh, w_hhh_hhh, 
                                                pnum, hnum, 
                                                sparse)
        for i in range(len(dum_two_body)):
            two_body_int[i] += dum_two_body[i]
        vacEn = get_ref_energy(f_hh, v_hhhh, w_hhh_hhh)
    else:
        vacEn = get_ref_energy(f_hh, v_hhhh, None)

    return vacEn, fock_mats, two_body_int

def get_norm_ordered_ham(thisL, holes, myTkin, mycontact, my3body=None, sparse=True, NO2B = True):
    """
    Takes all the necessary parameters to generate the fock matrices, two and three body interactions to perform CCM
    
    :param thisL:     number of lattice sites in each direction
    :type thisL:      int
    :param holes:     list of holes in the format [x, y, z, tz+0.5, sz+0.5]. To generate these, see :meth:`lattice.makeState`
    :type holes:      list[list[int]]
    :param myTkin:    list of one-body matrix elements [[p,q,T]...]
    :type myTkin:     list[list[int, int, float]]
    :param mycontact: list of two-body matrix elements [[p,q,r,s,V]...]                                                                                        
    :type mycontact:  list[list[int, int, int, int, float]]
    :param my3body:   list of two-body matrix elements [[p,q,r,s,u,v,W]...]                                                                                      
    :type my3body:    list[list[int, int, int, int, int, int, float]]
    :param sparse:    Optional whether or not to store v_pppp and v_ppph as a sparse array or not
    :type sparse:     bool
    :param NO2B:      whether or not to apply the normal-order two-body approximation, i.e. to return None
                      for the three body interaction
    :type NO2B:       bool
    :return:        The reference energy and three lists, a list of the three fock matrices in the order [f_pp, f_ph, f_hh], 
                    all of the two body interactions in the order [v_pppp, v_ppph, v_pphh, v_phph, 
                    v_phhh, v_hhhh], and all of the three body interactions in the order [w_ppp_pph, 
                    w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, w_pph_hhh, w_phh_phh, w_phh_hhh, 
                    w_hhh_hhh], or [] if my3body = None or NO2B = True
    :rtype:         float, list[numpy array], list[numpy array], list[numpy array]
    """
    hole, part = lat.states2PHSpace(holes, thisL)
    hnum = len(hole)
    pnum = len(part)

    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = get_all_interactions(part,hole,mycontact,sparse)
    
    f_pp, f_ph, f_hh = get_fock_matrices(part, hole, myTkin,v_phph, v_phhh, v_hhhh)

    three_body_int = None
    #getting the 3 body interactions if needed and adding the effective contribution to the fock matrices and two body interactions
    if my3body is not None:
        w_ppp_pph, w_ppp_phh, w_pph_pph, \
        w_ppp_hhh, w_pph_phh, w_pph_hhh, \
        w_phh_phh, w_phh_hhh, w_hhh_hhh = tbu.get_3NF(part, hole, my3body)

        # Include three-body contribution to Fock matrix
        dum_fpp, dum_fph, dum_fhh = tbu.get_3NF_fock(hnum, pnum, w_phh_phh, w_phh_hhh, w_hhh_hhh)
        
        f_pp += dum_fpp 
        f_ph += dum_fph
        f_hh += dum_fhh
        
        # Include three-body contribution to NO2B matrix elements
        dum_two_body = tbu.get_3NF_tbme(w_pph_pph, w_pph_phh, w_pph_hhh, 
                                        w_phh_phh, w_phh_hhh, w_hhh_hhh, 
                                        pnum, hnum, sparse)
        
        v_pppp += dum_two_body[0]
        v_ppph += dum_two_body[1]
        v_pphh += dum_two_body[2]
        v_phph += dum_two_body[3]
        v_phhh += dum_two_body[4]
        v_hhhh += dum_two_body[5]
        
        vacEn = get_ref_energy(f_hh, v_hhhh, w_hhh_hhh)
    else:
        vacEn = get_ref_energy(f_hh, v_hhhh, None)

    NO2B_stuff = vacEn, [f_pp, f_ph, f_hh], [v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh] 

    if NO2B or my3body is None:
        res = NO2B_stuff
    else:
        three_body_int = [w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh,
                          w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh]
        res = NO2B_stuff, three_body_int

    return res

