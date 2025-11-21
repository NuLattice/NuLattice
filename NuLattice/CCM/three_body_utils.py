"""
Provides functions for all of the 3 body interactions
"""
import numpy as np

def get_3NF(part, hole, my3body):
    """
    This routine takes the relatively small number of three-body matrix elements in mycontact
    and sorts them into the four-indexed interaction tensors. It also anti-symmetrizes the latter
    whenin and out indices run over the same set of particle/hole indices.
    The whole thing is a bit tedious but much faster than the function load2bme

    :param part:    list of particle-space indices
    :type part:     list[int]
    :param hole:    list of hole-space indices
    :type hole:     list[int]
    :param my3body: list of three-body matrix elements
    :type my3body:  list[(int, int, int, int, int, int, float)]
    :return:    w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, 
                w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh as lists of nonzeros  
    :rtype:     9 list[(int, int, int, int, int, int, float)]
    """
    pnum = len(part)
    hnum = len(hole)
    
    vals     = range(hnum)
    lookup_h = dict(zip(hole,vals))
    vals     = range(pnum)
    lookup_p = dict(zip(part,vals))
    
    w_ppp_pph = []
    w_ppp_phh = []
    w_pph_pph = []
    w_ppp_hhh = []
    w_pph_phh = []
    w_pph_hhh = []
    w_phh_phh = []
    w_phh_hhh = []
    w_hhh_hhh = []
    
    for [i1, i2, i3, i4, i5, i6, val] in my3body:
        # i1<i2<i3 and i4<i5<i6 is stored only in my3body;
        # we have to populate all permutations 
        
        iket = [i1, i2, i3]
        ibra = [i4, i5, i6]

        ket=[]
        bra=[]
        for ii in iket:
            if ii in hole:
                ket.append("h")
            else:
                ket.append("p")
      
        for ii in ibra:
            if ii in hole:
                bra.append("h")
            else:
                bra.append("p")

        ket_char = tuple(ket)
        bra_char = tuple(bra)
        
        ket, sign_ket, ket_indx = order_state(ket_char, lookup_p, lookup_h, i1, i2, i3)
        bra, sign_bra, bra_indx = order_state(bra_char, lookup_p, lookup_h, i4, i5, i6)

        [a, b, c] = ket_indx
        [d, e, f] = bra_indx
        
        if ket == ("p", "p", "p"):
            ket_perms = [[a, b, c], [b, a, c], [c, b, a], [a, c, b], [b, c, a], [c, a, b]]
            ket_signs = [1, -1, -1, -1, 1, 1]
            
            if bra == ("p", "p", "h"):
                vint = w_ppp_pph
                bra_perms = [[d, e, f], [e, d, f]]
                bra_signs = [1, -1]
                
            elif bra == ("p", "h", "h"):
                vint = w_ppp_phh
                bra_perms = [[d, e, f], [d, f, e]]
                bra_signs = [1, -1]
                        
            elif bra == ("h", "h", "h"):
                vint = w_ppp_hhh
                bra_perms = [[d, e, f], [e, d, f], [f, e, d], [d, f, e], [e, f, d], [f, d, e]]
                bra_signs = [1, -1, -1, -1, 1, 1]
            else:
                continue
                
        elif ket == ("p", "p", "h"):
            ket_perms = [[a, b, c], [b, a, c]]
            ket_signs = [1, -1]
            
            if bra == ("p", "p", "h"):
                vint = w_pph_pph
                bra_perms = [[d, e, f], [e, d, f]]
                bra_signs = [1, -1]

            elif bra == ("p", "h", "h"):
                vint = w_pph_phh
                bra_perms = [[d, e, f], [d, f, e]]
                bra_signs = [1, -1]
                        
            elif bra == ("h", "h", "h"):
                vint = w_pph_hhh
                bra_perms = [[d, e, f], [e, d, f], [f, e, d], [d, f, e], [e, f, d], [f, d, e]]
                bra_signs = [1, -1, -1, -1, 1, 1]
            else:
                continue

        elif ket == ("p", "h", "h"):
            ket_perms = [[a, b, c], [a, c, b]]
            ket_signs = [1, -1]
            
            if bra == ("p", "h", "h"):
                vint = w_phh_phh
                bra_perms = [[d, e, f], [d, f, e]]
                bra_signs = [1, -1]
                
            elif bra == ("h", "h", "h"):
                vint = w_pph_hhh
                bra_perms = [[d, e, f], [e, d, f], [f, e, d], [d, f, e], [e, f, d], [f, d, e]]
                bra_signs = [1, -1, -1, -1, 1, 1]
            else:
                continue
    
        elif ket == ("h", "h", "h"):
            ket_perms = [[a, b, c], [b, a, c], [c, b, a], [a, c, b], [b, c, a], [c, a, b]]
            ket_signs = [1, -1, -1, -1, 1, 1]
            
            if bra == ("h", "h", "h"):
                vint = w_hhh_hhh
                bra_perms = [[d, e, f], [e, d, f], [f, e, d], [d, f, e], [e, f, d], [f, d, e]]
                bra_signs = [1, -1, -1, -1, 1, 1]
            else:
                continue
                    
        else:
            continue # the matrix element is not needed (we exploit that 3NF is real symmetric)
            
        indices = []
        signs = []
        for ii, ele_k in  enumerate(ket_perms):
            for jj, ele_b in enumerate(bra_perms):
                indices.append(ele_k + ele_b)
                signs.append(ket_signs[ii] * bra_signs[jj])
      
                
        for i, indx in enumerate(indices):
            sign = signs[i]
            [a1, a2, a3, a4, a5, a6] = indx
            vint.append([a1, a2, a3, a4, a5, a6, val * sign_ket * sign_bra * sign])
            
    return w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh

def order_state(ket,lookup_p,lookup_h, i1, i2, i3):
    """
    orders tuple into right order, i.e. "p" before "h"
    this function is used by the function get_3NF

    :param ket:         ket to be used in finding the right particle hole index
    :type ket:          (str, str, str)
    :param lookup_p:    dictionary to lookup the index of particles
    :type lookup_p:     dict(int, int)
    :param lookup_h:    dictionary to lookup the index of holes
    :type lookup_h:     dict(int, int)
    :param i1:          index of first particle/hole
    :type i1:           int
    :param i2:          index of second particle/hole
    :type i2:           int
    :param i3:          index of third particle/hole
    :type i3:           int
    :return:            the correctly ordered result looking like ket,
                        the sign of the permutation that achieved this,
                        and the list of three single-particle indices
    :rtype:             tuple(str, str, str), float, list[(int, int, int)]
    """
    res = ket
    if ket == ("p","p","p"):
        sign_ket = 1.0
        a = lookup_p.get(i1)
        b = lookup_p.get(i2)
        c = lookup_p.get(i3)
    elif ket == ("h","h","h"):
        sign_ket = 1.0
        a = lookup_h.get(i1)
        b = lookup_h.get(i2)
        c = lookup_h.get(i3)
    elif ket == ("p","p","h"):
        sign_ket = 1.0
        a = lookup_p.get(i1)
        b = lookup_p.get(i2)
        c = lookup_h.get(i3)
    elif ket == ("p","h","p"):
        sign_ket = -1.0
        a = lookup_p.get(i1)
        c = lookup_h.get(i2)
        b = lookup_p.get(i3)
        res = ("p","p","h")
    elif ket == ("h","p","p"):
        sign_ket = -1.0
        c = lookup_h.get(i1)
        b = lookup_p.get(i2)
        a = lookup_p.get(i3)
        res = ("p","p","h")
    elif ket == ("p","h","h"):
        sign_ket = 1.0
        a = lookup_p.get(i1)
        b = lookup_h.get(i2)
        c = lookup_h.get(i3)
    elif ket == ("h","p","h"):
        sign_ket = -1.0
        b = lookup_h.get(i1)
        a = lookup_p.get(i2)
        c = lookup_h.get(i3)
        res = ("p","h","h")
    elif ket == ("h","h","p"):
        sign_ket = -1.0
        c = lookup_h.get(i1)
        b = lookup_h.get(i2)
        a = lookup_p.get(i3)
        res = ("p","h","h")
        
    return res, sign_ket, [a, b, c]


def get_3NF_Eref(w_hhh_hhh):
    """
    returns normal-ordering contributions to the reference energy 

    :param w_hhh_hhh:   nonzero elements of the three body interaction W^{ijk}_{lmn}
    :type w_hhh_hhh:    list[(int, int, int, int, int, int, float)]
    :return:            contribution to the reference energy from the normal ordered three nucleon force
    :rtype:             float
    """
    Eref = 0.0
    for ele in w_hhh_hhh:
        [m, i, j, n, k, l, val] = ele
#        if (m, n, i) == (n, k, l):
        if m == n and i == k and j == l: 
            Eref += val/6.0
            
    return Eref
    

def get_3NF_fock(hnum, pnum, w_phh_phh, w_phh_hhh, w_hhh_hhh):
    """
    gets the normal-ordering contributions of the three-body potential to the Fock matrix

    :param hnum:    number of hole states
    :type hnum:     int
    :param pnum:    number of particle states
    :type pnum:     int
    :param w_phh_phh:   nonzero elements of the three body interaction matrix W^{aij}_{bkl}
    :type w_phh_phh:    list[(int, int, int, int, int, int, float)]
    :param w_phh_hhh:   nonzero elements of the three body interaction matrix W^{aij}_{klm}
    :type w_phh_hhh:    list[(int, int, int, int, int, int, float)]
    :param w_hhh_hhh:   nonzero elements of the three body interaction matrix W^{ijk}_{lmn}
    :type w_hhh_hhh:    list[(int, int, int, int, int, int, float)]
    :return:            contributions to the normal ordered one body matrices
    :rtype:             numpy array, numpy array, numpy array
    """
    f_pp = np.zeros((pnum, pnum))
    f_ph = np.zeros((pnum, hnum))
    f_hh = np.zeros((hnum, hnum))
    
    for ele in w_phh_phh:
        [a, i, j, b, k, l, val] = ele
        if i == k and j == l:
            f_pp[a,b] += 0.5*val
            
    for ele in w_phh_hhh:
        [a, i, j, m, k, l, val] = ele
        if i == k and j == l:
            f_ph[a,m] += 0.5*val
            
    for ele in w_hhh_hhh:
        [n, i, j, m, k, l, val] = ele
        if i == k and j == l:
            f_hh[n,m] += 0.5*val
            
    return f_pp, f_ph, f_hh


def get_3NF_tbme(w_pph_pph, w_pph_phh, w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh, 
                 pnum, hnum, sparse_pppp = True, sparse_ppph = True):
    """
    Finds the normal-ordering contributions of the three-body potential to the two-body matrix elements

    :param w_pph_pph:   nonzero elements of the three body interaction matrix W^{abi}_{cdj}
    :type w_pph_pph:    list[(int, int, int, int, int, int, float)]
    :param w_pph_phh:   nonzero elements of the three body interaction matrix W^{acj}_{bkl}
    :type w_pph_phh:    list[(int, int, int, int, int, int, float)]
    :param w_phh_phh:   nonzero elements of the three body interaction matrix W^{aij}_{bkl}
    :type w_phh_phh:    list[(int, int, int, int, int, int, float)]
    :param w_phh_hhh:   nonzero elements of the three body interaction matrix W^{aij}_{klm}
    :type w_phh_hhh:    list[(int, int, int, int, int, int, float)]
    :param w_hhh_hhh:   nonzero elements of the three body interaction matrix W^{ijk}_{lmn}
    :type w_hhh_hhh:    list[(int, int, int, int, int, int, float)]
    :param hnum:    number of hole states
    :type hnum:     int
    :param pnum:    number of particle states
    :type pnum:     int
    :param sparse_pppp: Optional; whether or not v_pppp should be stored as sparse or not
    :type sparse_pppp:  bool
    :param sparse_ppph: Optional; whether or not v_ppph should be stored as sparse or not
    :type sparse_ppph:  bool
    :return:            contributions to the normal ordered two body matrices
    :rtype:             numpy array, numpy array, numpy array, numpy array, numpy array, numpy array
    """
    v_phph = np.zeros((pnum, hnum, pnum, hnum))
    v_pphh = np.zeros((pnum, pnum, hnum, hnum))
    v_phhh = np.zeros((pnum, hnum, hnum, hnum))
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum)) 
    
    if sparse_pppp:
        v_pppp = []
    else:
        v_pppp = np.zeros((pnum, pnum, pnum, pnum))
        
    if sparse_ppph:
        v_ppph = []
    else:
        v_ppph = np.zeros((pnum, pnum, pnum, hnum))
                
    for ele in w_pph_pph:
        [a, b, i, c, d, j, val] = ele
        if i == j:
            if sparse_pppp:
                v_pppp.append([a, b, c, d, val])
            else:
                v_pppp[a, b, c, d] += val
       
    for ele in w_pph_phh:
        [a, b, i, c, k, j, val] = ele
        if i == j:
            if sparse_pppp:
                v_ppph.append([a, b, c, k, val])
            else:
                v_ppph[a, b, c, k] += val

    for ele in w_pph_hhh:
        [a, b, i, m, k, j, val] = ele
        if i == j:
            v_pphh[a, b, m, k] += val
            
    for ele in w_phh_phh:
        [a, n, i, b, k, j, val] = ele
        if i == j:
            v_phph[a, n, b, k] += val
            
    for ele in w_phh_hhh:
        [a, n, i, l, k, j, val] = ele
        if i == j:
            v_phhh[a, n, l, k] += val
            
    for ele in w_hhh_hhh:
        [m, n, i, l, k, j, val] = ele
        if i == j:
            v_hhhh[m, n, l, k] += val
            
    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh
