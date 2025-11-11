"""
module provides functions to build a many-body basis and to construct
Hamiltonian matrices stored in compressed storage row csr format.
"""
__authors__   =  "Thomas Papenbrock"
__credits__   =  ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__   = "BSD-3-Clause"
__date__      = "2025-07-26"

from itertools import combinations
import numpy as np
from scipy import sparse as sparse


def get_many_body_states(basis, num_part, total_tz=None, total_sz=None):
    """
    returns dictionary of many-body states

    :param basis:     the single-particle basis
    :type basis:      list[list[int, int, int, int, int], [ ...]]
    :param num_part:  number of fermions
    :type num_part:   int
    :param total_tz: total z-component of isospin (twice its value) 
    :type total_tz:   int
    :param total_sz:  total z-component of spin (twice its value)
    :type total_sz:   int
    :return:          a dictionary where keys are tuples of single-particle
                      states and values are the indices of that many-body
                      state; this eerves as a lookup table.
    :rtype:           dict(tuple(int, int, int, ...): int)
    """

    nstat = len(basis)
    iter_states = combinations(range(nstat), num_part)
    # yields all combinations (nstat choose num_part) as an iterator of tuples
    # iterators can only be used once

    many_body_bas = []
    for values in iter_states:
        if total_tz is not None:
            tz = 0
            for i in range(num_part):
                tz += basis[values[i]][3]
            tz = 2*tz - num_part
            if tz != total_tz:
                continue
        if total_sz is not None:
            sz = 0
            for i in range(num_part):
                sz += basis[values[i]][4]
            sz = 2*sz - num_part
            if sz != total_sz:
                continue

        many_body_bas.append(values)

    dim  = len(many_body_bas)

    vals = range(dim)
    return dict(zip(many_body_bas, vals))

def get_csr_matrix_scalar_op(lookup, operator, num_sp_stat):
    """
    returns a scalar operator as a CSR matrix using the lookup
    dictionary. The few-body basis can only have A=2, 3, or 4
    particles, and the rank of the operator can only be 1, 2, or 3.
    
    :param lookup:      dictionary of A-body states
    :type lookup:       dict(tuple(int,int,...): int)
    :param operator:    list of matrix elements of the few-body operator
    :type operator:     list[list[int, int, int, int, float], [...]]
    :param num_sp_stat: number of single-particle states
    :type num_sp_stat:  int
    :return:            csr matrix of the operator
    :rtype:             scipy.sparse csr_matrix
    """

    nstat = num_sp_stat
    rank_op = (len(operator[0]) - 1)//2
    first_key = next(iter(lookup))
    num_part_basis = len(first_key)
    dim = len(lookup)

    if rank_op == 1:
        if num_part_basis == 2:
            res = get_csr_1b_op_in_2b_basis(lookup,operator,nstat)
        elif num_part_basis == 3:
            res = get_csr_1b_op_in_3b_basis(lookup,operator,nstat)
        elif num_part_basis == 4:
            res = get_csr_1b_op_in_4b_basis(lookup,operator,nstat)
        else:
            res = None
    elif rank_op == 2:
        if num_part_basis == 2:
            res = get_csr_2b_op_in_2b_basis(lookup,operator)
        elif num_part_basis == 3:
            res = get_csr_2b_op_in_3b_basis(lookup,operator,nstat)
        elif num_part_basis == 4:
            res = get_csr_2b_op_in_4b_basis(lookup,operator,nstat)
        else:
            res = None
    elif rank_op == 3:
        if num_part_basis == 3:
            res = get_csr_3b_op_in_3b_basis(lookup,operator)
        elif num_part_basis == 4:
            res = get_csr_3b_op_in_4b_basis(lookup,operator,nstat)
        else:
            res = None
    else:
        res = None

    return res

def fill_1b_op_in_2b_basis(lookup, operator, nstat):
    """
    for a 2-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 1-body operator

    :param lookup:    dictionary of two-body states
    :type lookup:     dict(tuple(int,int): int)
    :param operator:  one-body operator as list of [row, col, value]
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]
    """
    op_dat = []
    op_row = []
    op_col = []
    for matele in operator:
        [i1, j1, val] = matele
        for p2 in range(nstat): # get second particle
            if p2 in [i1, j1]: #Pauli says no
                continue
            if i1 < p2: # i1, p2
                stat1 = (i1, p2)
                sign1 = 1.0
            elif p2 < i1:
                stat1 = (p2, i1)
                sign1 = -1.0

            ii = lookup.get(stat1)
            if ii is None: # state nor found, wrong quantum numbers
                continue
            if j1 < p2: # j1, p2
                stat2 = (j1, p2)
                sign2 = 1.0
            elif p2 < j1: # p2, j1
                stat2 = (p2, j1)
                sign2 = -1.0

            jj = lookup.get(stat2)
            if jj is None: # state nor found, wrong quantum numbers
                continue

            op_dat.append(val*sign1*sign2)
            op_row.append(ii)
            op_col.append(jj)

    return op_dat, op_row, op_col


def fill_2b_op_in_2b_basis(lookup,operator):
    """
    for a 2-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 2-body operator

    :param lookup:    dictionary of two-body states
    :type lookup:     dict(tuple(int,int): int)
    :param operator:  two-body operator as list of [p, q, r, s, value] where p, q, r, s
                      are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]
    """
    op_dat = []
    op_row = []
    op_col = []
    for tbme in operator:
        [a, b, c, d, val] = tbme
        ii = lookup.get((a,b))
        jj = lookup.get((c,d))
        if ii is None or jj is None: # not found
            continue

        op_dat.append(val)
        op_row.append(ii)
        op_col.append(jj)

    return op_dat, op_row, op_col


def fill_1b_op_in_3b_basis(lookup,operator,nstat):
    """
    for a 3-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 1-body operator

    :param lookup:    dictionary of three-body states
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  one-body operator as list of [p, q, value] where p, q
                      are one-body states 
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]
    """
    op_dat = []
    op_row = []
    op_col = []
    for matele in operator:
        [i1, j1, val] = matele
        for p2 in range(nstat): #get second particle
            if p2 in [i1, j1]: #Pauli says no
                continue
            for p3 in range(p2+1, nstat): #get third particle
                if p3 in [i1, j1]: #Pauli says no
                    continue

                if i1 < p2: #i1, p2, p3
                    stat1 = (i1, p2, p3)
                    sign1 = 1.0
                elif p2 < i1 and i1 < p3: #p2, i1, p3
                    stat1 = (p2, i1, p3)
                    sign1 = -1.0
                elif p3 < i1: #p2, p3, i1
                    stat1 = (p2, p3, i1)
                    sign1 = 1.0

                ii = lookup.get(stat1)
                if ii is None: # not found, wrong quantum numbers
                    continue

                if j1 < p2: #j1, p2, p3
                    stat2 = (j1, p2, p3)
                    sign2 = 1.0
                elif p2 < j1 and j1 < p3: #p2, j1, p3
                    stat2 = (p2, j1, p3)
                    sign2 = -1.0
                elif p3 < j1: #p2, p3, j1
                    stat2 = (p2, p3, j1)
                    sign2 = 1.0

                jj = lookup.get( stat2 )
                if jj is None: # not found, wrong quantum numbers
                    continue

                op_dat.append(val*sign1*sign2)
                op_row.append(ii)
                op_col.append(jj)

    return op_dat, op_row, op_col


def fill_2b_op_in_3b_basis(lookup,operator,nstat):
    """
    for a 3-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 2-body operator

    :param lookup:    dictionary of three-body states where keys are tuples (p q r) of one-body
                      states and values are the index of the corresponding three-body basis state
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  two-body operator as list of [p, q, r, s, value] where p, q, r, s
                      are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]    
    """

    op_dat = []
    op_row = []
    op_col = []
    for tbme in operator:
        [i1, i2, j1, j2, val] = tbme
        for p3 in range(nstat):
            if p3 in [i1, i2, j1, j2]: # Pauli forbids it
                continue

            if p3 < i1:
                stat1 = (p3, i1, i2)
                sign1 = 1.0
            elif p3 > i1 and p3 < i2:
                stat1 = (i1, p3, i2)
                sign1 = -1.0
            elif p3 > i2:
                stat1 = (i1, i2, p3)
                sign1 = 1.0

            ii = lookup.get(stat1)

            if ii is None: # not found, wrong quantum numbers
                continue

            if p3 < j1:
                stat2 = (p3, j1, j2)
                sign2 = 1.0
            elif p3 > j1 and p3 < j2:
                stat2 = (j1, p3, j2)
                sign2 = -1.0
            elif p3 > j2:
                stat2 = (j1, j2, p3)
                sign2 = 1.0

            jj = lookup.get(stat2)

            if jj is None:
                continue

            op_dat.append(val*sign1*sign2)
            op_row.append(ii)
            op_col.append(jj)

    return op_dat, op_row, op_col


def fill_3b_op_in_3b_basis(lookup,operator):
    """
    for a 3-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 3-body operator

    :param lookup:    dictionary of three-body states where keys are tuples (p q r) of one-body
                      states and values are the index of the corresponding three-body basis state
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  three-body operator as list of [p, q, r, s, u, v, value] where p, q, r, s
                      u, v are one-body states 
    :type operator:   list[list[int,int,int,int,int,int,float]]
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]    
    """
    op_dat = []
    op_row = []
    op_col = []
    for nnn in operator:
        [i1, i2, i3, j1, j2, j3, val] = nnn
        ii = lookup.get( (i1, i2, i3) )
        jj = lookup.get( (j1, j2, j3) )

        if ii is None or jj is None:
            continue # quantum numbers do not match

        op_dat.append(val)
        op_row.append(ii)
        op_col.append(jj)

    return op_dat, op_row, op_col

def fill_1b_op_in_4b_basis(lookup,operator,nstat):
    """
    for a 4-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 1-body operator

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  one-body operator as list of [p, q, value] where p, q
                      are one-body states 
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]
    """
    op_dat = []
    op_row = []
    op_col = []
    for matele in operator:
        [i1, j1, val] = matele
        for p2 in range(nstat): # get second particle
            if p2 in [i1, j1]: #Pauli says no
                continue
            for p3 in range(p2+1, nstat): # get third particle
                if p3 in [i1, j1]: #Pauli says no
                    continue
                for p4 in range(p3+1, nstat): # get third particle
                    if p4 in [i1, j1]: #Pauli says no
                        continue

                    if i1 < p2: # i1, p2, p3, p4
                        stat1 = (i1, p2, p3, p4)
                        sign1 = 1.0
                    elif p2 < i1 and i1 < p3: # p2, i1, p3, p4
                        stat1 = (p2, i1, p3, p4)
                        sign1 = -1.0
                    elif p3 < i1 and i1 < p4: # p2, p3, i1, p4
                        stat1 = (p2, p3, i1, p4)
                        sign1 = 1.0
                    elif p4 < i1: # p2, p3, p4, i1
                        stat1 = (p2, p3, p4, i1)
                        sign1 = -1.0

                    ii = lookup.get(stat1)
                    if ii is None: # not found, wrong quantum numbers
                        continue

                    if j1 < p2: # j1, p2, p3, p4
                        stat2=(j1,p2,p3,p4)
                        sign2=1.0
                    elif p2 < j1 and j1 < p3: # p2, j1, p3, p4
                        stat2=(p2,j1,p3,p4)
                        sign2=-1.0
                    elif p3 < j1 and j1 < p4: # p2, p3, j1
                        stat2=(p2,p3,j1,p4)
                        sign2=1.0
                    elif p4 < j1: # p2, p3, p4, j1
                        stat2=(p2,p3,p4,j1)
                        sign2=-1.0

                    jj = lookup.get( stat2 )
                    if jj is None: # not found, wrong quantum numbers
                        continue

                    op_dat.append(val*sign1*sign2)
                    op_row.append(ii)
                    op_col.append(jj)

    return op_dat, op_row, op_col

def fill_2b_op_in_4b_basis(lookup,operator,nstat):
    """
    for a 4-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 2-body operator

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  two-body operator as list of [p, q, r, s, value] where p, q, r, s
                      are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]    
    """
    op_dat=[]
    op_row=[]
    op_col=[]
    for tbme in operator:
        [i1, i2, j1, j2, val] = tbme
        for p3 in range(nstat):
            if p3 in [i1,i2,j1,j2]: # Pauli forbids it
                continue
            for p4 in range(p3+1,nstat):
                if p4 in [i1,i2,j1,j2]: # Pauli forbids it
                    continue

                if p4 < i1: # p3, p4, i1, i2
                    stat1=(p3,p4,i1,i2)
                    sign1=1.0
                elif p3 < i1 and i1 < p4 and p4 < i2: # p3, i1, p4, i2
                    stat1=(p3,i1,p4,i2)
                    sign1=-1.0
                elif p3 < i1 and i2 < p4: # p3, i1, i2, p4
                    stat1=(p3,i1,i2,p4)
                    sign1=1.0
                elif i1 < p3 and p3 < i2 and i2 < p4: #i1, p3, i2, p4
                    stat1=(i1,p3,i2,p4)
                    sign1=-1.0
                elif i1 < p3 and p4 < i2: # i1, p3, p4, i2
                    stat1=(i1,p3,p4,i2)
                    sign1=1.0
                elif i2 < p3: # i1, i2, p3, p4
                    stat1=(i1,i2,p3,p4)
                    sign1=1.0

                ii = lookup.get( stat1 )
                if ii is None: # not found, wrong quantum numbers
                    continue

                if p4 < j1:
                    stat2=(p3,p4,j1,j2)
                    sign2=1.0
                elif p3 < j1 and j1 < p4 and p4 < j2:
                    stat2=(p3,j1,p4,j2)
                    sign2=-1.0
                elif p3 < j1 and j2 < p4:
                    stat2=(p3,j1,j2,p4)
                    sign2=1.0
                elif j1 < p3 and p3 < j2 and j2 < p4:
                    stat2=(j1,p3,j2,p4)
                    sign2=-1.0
                elif i1 < p3 and p4 < j2:
                    stat2=(j1,p3,p4,j2)
                    sign2=1.0
                elif j2 < p3:
                    stat2=(j1,j2,p3,p4)
                    sign2=1.0

                jj = lookup.get( stat2 )
                if jj is None:
                    continue

                op_dat.append(val*sign1*sign2)
                op_row.append(ii)
                op_col.append(jj)

    return op_dat, op_row, op_col


def fill_3b_op_in_4b_basis(lookup,operator,nstat):
    """
    for a 4-body system this function  returns a list of matrix elements,
    a list of row indices, and a list of column indices of the 3-body operator

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  three-body operator as list of [p, q, r, s, u, v, value] where p, q, r, s,
                      u, v are one-body states 
    :type operator:   list[list[int,int,int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          operator matrix elements as three lists op_dat, op_row, op_col
    :rtype:           list[float], list[int], list[int]    
    """
    op_dat=[]
    op_row=[]
    op_col=[]
    for nnn in operator:
        [i1,i2,i3,j1,j2,j3, val] = nnn
        for p4 in range(nstat):
            if p4 in [i1,i2,i3,j1,j2,j3]: # Pauli forbids it
                continue

            if p4 < i1:# p4, i1, i2, i3
                stat1=(p4,i1,i2,i3)
                sign1=1.0
            elif i1 < p4 and p4 < i2:
                stat1=(i1,p4,i2,i3)
                sign1=-1.0
            elif i2 < p4 and p4 < i3:
                stat1=(i1,i2,p4,i3)
                sign1=1.0
            elif i3 < p4:
                stat1=(i1,i2,i3,p4)
                sign1=-1.0

            ii = lookup.get( stat1 )
            if ii is None: # not found, wrong quantum numbers
                continue

            if p4 < j1: # p4, j1, j2, j3
                stat2=(p4,j1,j2,j3)
                sign2=1.0
            elif j1 < p4 and p4 < j2:
                stat2=(j1,p4,j2,j3)
                sign2=-1.0
            elif j2 < p4 and p4 < j3:
                stat2=(j1,j2,p4,j3)
                sign2=1.0
            elif j3 < p4:
                stat2=(j1,j2,j3,p4)
                sign2=-1.0

            jj = lookup.get( stat2 )
            if jj is None: # not found, wrong quantum numbers
                continue

            op_dat.append(val*sign1*sign2)
            op_row.append(ii)
            op_col.append(jj)

    return op_dat, op_row, op_col


## shift operator (to check which state is translationally invariant)

def num_permutations(my_list):
    """
    returns the number of permutations needed to bring a list into order

    :param mylist:  a list of integers
    :type mylist:   list[int]
    :return:        number of permutations needed to bring a list into order
    :rtype:         int
    """
    num=len(my_list)
    my_l = copy.deepcopy(my_list)
    res=0
    for i in range(num):
        smallest = min(my_l)  # Get the smallest element
        position = my_l.index(smallest)
        my_l.pop(position)
        res=res+position
    return res


def fill_shift_op(direc, lookup, myL, spin=2, isospin=2):
    """
    for a system defined via the lookup table this function  returns
    a list of matrix elements, a list of row indices, and a list of column indices
    of the shift operator that moves the state by a single lattice unit into direction

    :param direc:     direction of the shift, must be 1, 2, or 3
    :type direc:      int
    :param lookup:    dictionary of few-body states where keys are tuples of one-body
                      states and values are the index of the corresponding few-body basis state
    :type lookup:     dict(tuple(int,int,...): int)
    :param myL:       number of lattcie sites in each dimension
    :type myL:        int
    :param spin:      number of spin components
    :type spin:       int
    :param isospin:   number of isospin components
    :type isospin:    int
    :return:          a list of matrix elements, a list of row indices, and a list of column indices
    :rtype:           list[float], list[int], list[int]
    """
    if direc not in [1,2,3]:
        print(" Direction needs to be 1, 2, or 3. Choosing 3")
        direc=3

    myshift = spin*isospin*myL**(3-direc)
    unit = spin*isospin*myL**(4-direc)
    mat_dat=[]
    mat_row=[]
    mat_col=[]
    for key, value in lookup.items():
        ii  = np.array(key,dtype=int)
        offset = ii//unit

        rem = ( ii + myshift  )%unit #remainder

        new = offset*unit + rem
        sortnew = np.sort(new)
        perm=num_permutations(new.tolist())

        sign = (-1.0)**perm

        newkey = tuple( sortnew )

        ind = lookup.get(newkey)
        if ind is None:
            continue
        mat_dat.append(sign)
        mat_row.append(ind)
        mat_col.append(value)

    return  mat_dat, mat_row, mat_col


def get_csr_1b_op_in_2b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 1-body operator into
    a 2-body basis
    
    :param lookup:    dictionary of two-body states where keys are tuples (p q) of one-body
                      states and values are the index of the corresponding two-body basis state
    :type lookup:     dict(tuple(int,int): int)
    :param operator:  one-body operator as list of [p, q, value] where p, q, are one-body states 
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim2 = len(lookup)
    op_dat, op_row, op_col = fill_1b_op_in_2b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim2,dim2) )
    return res

def get_csr_2b_op_in_2b_basis(lookup,operator):
    """
    returns a compressed sparse row 'csr' matrix of a 2-body operator into
    a 2-body basis

    :param lookup:    dictionary of two-body states where keys are tuples (p q) of one-body
                      states and values are the index of the corresponding two-body basis state
    :type lookup:     dict(tuple(int,int): int)
    :param operator:  one-body operator as list of [p, q, r, s, value] where p, q, r, s are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim2 = len(lookup)
    op_dat, op_row, op_col = fill_2b_op_in_2b_basis(lookup,operator)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim2,dim2) )
    return res

def get_csr_1b_op_in_3b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 1-body operator into
    a 3-body basis

    :param lookup:    dictionary of three-body states where keys are tuples (p q r) of one-body
                      states and values are the index of the corresponding three-body basis state
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  one-body operator as list of [p, q, value] where p, q, are one-body states 
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim3 = len(lookup)
    op_dat, op_row, op_col = fill_1b_op_in_3b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim3,dim3) )
    return res

def get_csr_2b_op_in_3b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 2-body operator into
    a 3-body basis

    :param lookup:    dictionary of three-body states where keys are tuples (p q r) of one-body
                      states and values are the index of the corresponding three-body basis state
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  two-body operator as list of [p,q,r,s,value] where p,q,r,s are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim3 = len(lookup)
    op_dat, op_row, op_col = fill_2b_op_in_3b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim3,dim3) )
    return res

def get_csr_3b_op_in_3b_basis(lookup,operator):
    """
    returns a compressed sparse row 'csr' matrix of a 3-body operator into
    a 3-body basis

    :param lookup:    dictionary of three-body states where keys are tuples (p q r) of one-body
                      states and values are the index of the corresponding two-body basis state
    :type lookup:     dict(tuple(int,int,int): int)
    :param operator:  one-body operator as list of [p,q,r,s,u,v, value] where p,q,r,s,u,v are one-body states 
    :type operator:   list[list[int,int,int,int,int,int,float]]
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim3 = len(lookup)
    op_dat, op_row, op_col = fill_3b_op_in_3b_basis(lookup,operator)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim3,dim3) )
    return res

def get_csr_1b_op_in_4b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 1-body operator into
    a 4-body basis

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  one-body operator as list of [p,q,value] where p,q are one-body states 
    :type operator:   list[list[int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim4 = len(lookup)
    op_dat, op_row, op_col = fill_1b_op_in_4b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim4,dim4) )
    return res

def get_csr_2b_op_in_4b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 2-body operator into
    a 4-body basis

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  two-body operator as list of [p,q,r,s,value] where p,q,r,s are one-body states 
    :type operator:   list[list[int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim4 = len(lookup)
    op_dat, op_row, op_col = fill_2b_op_in_4b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim4,dim4) )
    return res

def get_csr_3b_op_in_4b_basis(lookup,operator,nstat):
    """
    returns a compressed sparse row 'csr' matrix of a 3-body operator into
    a 4-body basis

    :param lookup:    dictionary of four-body states where keys are tuples (p q r s) of one-body
                      states and values are the index of the corresponding four-body basis state
    :type lookup:     dict(tuple(int,int,int,int): int)
    :param operator:  two-body operator as list of [p,q,r,s,u,v,value] where p,q,r,s,u,v are one-body states 
    :type operator:   list[list[int,int,int,int,int,int,float]]
    :param nstast:    number of single-particle states
    :type nstat:      int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim4 = len(lookup)
    op_dat, op_row, op_col = fill_3b_op_in_4b_basis(lookup,operator,nstat)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim4,dim4) )
    return res

def get_shift_op(direc, lookup, myL, spin=2, isospin=2):
    """
    returns a compressed sparse row 'csr' matrix of the shift operator into
    a few-body basis
    
    :param direc:     direction of the shift, must be 1, 2, or 3
    :type direc:      int
    :param lookup:    dictionary of few-body states where keys are tuples of one-body
                      states and values are the index of the corresponding few-body basis state
    :type lookup:     dict(tuple(int,int,...): int)
    :param myL:       number of lattcie sites in each dimension
    :type myL:        int
    :param spin:      number of spin components
    :type spin:       int
    :param isospin:   number of isospin components
    :type isospin:    int
    :return:          csr matrix of the operator
    :rtype:           scipy.sparse.csr_matrix
    """
    dim = len(lookup)
    op_dat, op_row, op_col = fill_shift_op(direc, lookup, myL, spin=spin, isospin=isospin)
    res = sparse.csr_matrix( (op_dat, (op_row, op_col)), shape=(dim,dim) )
    return res


def csr_matrix_tolist_2body(op_csr,lookup2b):
    """
    converts a scipy.sparse.csr_matrix into a list of elements [p,q,r,s,value]
    
    :param op_csr:   csr_matrix from scipy.sparse
    :type op_csr:    csr_matrix from scipy.sparse
    :param lookup2b: dictionary where keys are two-body state tuples (p,q) and values are indices 
    :type lookup2b:  dictionary with entries {(int,int): int}
    :return:         list [[p,q,r,s,value], ] of two-body matrix elements
    :rtype:          list with elements [int,int,int,int, float]
    """
    keys=[]
    vals=[]
    
    for key, value in lookup2b.items():
        keys.append(key)
        vals.append(value)

    inverse_lookup = dict(zip(vals,keys))

    (rows, cols, vals) = sparse.find(op_csr)
    
#     vals = np.real(vals)
 
    op_list=[]
    for i, val in enumerate(vals):
        ii = rows[i]
        jj = cols[i]
        (a, b) = inverse_lookup[ii]
        (c, d) = inverse_lookup[jj]
        op_list.append([a,b,c,d,val])
    
    return op_list
    
def add_2body_ops(ops, my_basis, weights=None):
    """
    adds lists of sparse 2-body operators into a single sparse list
    
    :param ops:      list of lists [[p,q,r,s, val], ...] of two-body operators 
    :type ops:       [[[int,int,int,int, float], ...], ...]
    :param my_basis: list with elements [x,y,z,tauz,sz] that are single-particle states
    :type my_basis:  [int,int,int,int,int]
    :param weights:  array of weights each operator in ops will be multiplied with
    :type weights:   [float,float,...]
    :return:         list [[p,q,r,s, val], ...] of a single two-body operator
    :rtype:          [[int,int,int,int, float], ...]
    """
    # main idea: convert each list into csr_matrix, add them all up, and 
    # convert back into a single list 
    num_ops = len(ops)
    if weights is None:
        ww=np.ones(num_ops)
    else:
        ww=weights
        
    nbody=2
    lookup2b = get_many_body_states(my_basis, nbody)

    op_csr = ww[0]*get_csr_2b_op_in_2b_basis(lookup2b,ops[0])
    for i in range(1,num_ops):
        op_csr += ww[i]*get_csr_2b_op_in_2b_basis(lookup2b,ops[i])    
    
    return csr_matrix_tolist_2body(op_csr,lookup2b)

def csr_matrix_tolist_3body(op_csr,lookup3b):
    """
    converts a scipy.sparse.csr_matrix into a list of elements [p,q,r,s,u,v,value]
    
    :param op_csr:   csr_matrix from scipy.sparse
    :type op_csr:    csr_matrix from scipy.sparse
    :param lookup3b: dictionary where keys are two-body state tuples (p,q,r) and values are indices 
    :type lookup3b:  dictionary with entries {(int,int,int): int}
    :return:         list [[p,q,r,s,u,v, value], ] of two-body matrix elements
    :rtype:          list with elements [int,int,int, int,int,int, float]
    """
    keys=[]
    vals=[]
    
    for key, value in lookup3b.items():
        keys.append(key)
        vals.append(value)

    inverse_lookup = dict(zip(vals,keys))

    (rows, cols, vals) = sparse.find(op_csr)
    
#     vals = np.real(vals)
 
    op_list=[]
    for i, val in enumerate(vals):
        ii = rows[i]
        jj = cols[i]
        (a, b, c) = inverse_lookup[ii]
        (d, e, f) = inverse_lookup[jj]
        op_list.append([a,b,c,d,e,f,val])
    
    return op_list
    
def add_3body_ops(ops, my_basis, weights=None):
    """
    adds lists of sparse 3-body operators into a single sparse list
    
    :param ops:      list of lists [[p,q,r,s,u,v, val], ...] of three-body operators 
    :type ops:       [[[int,int,int,int,int,int, float], ...], ...]
    :param my_basis: list with elements [x,y,z,tauz,sz] that are single-particle states
    :type my_basis:  [int,int,int,int,int]
    :param weights:  array of weights each operator in ops will be multiplied with
    :type weights:   [float,float,...]
    :return:         list [[p,q,r,s,u,v, val], ...] of a single three-body operator
    :rtype:          [[int,int,int,int,int,int, float], ...]
    """
    # main idea: convert each list into csr_matrix, add them all up, and 
    # convert back into a single list 
    num_ops = len(ops)
    if weights is None:
        ww=np.ones(num_ops)
    else:
        ww=weights
        
    nbody=3
    lookup3b = get_many_body_states(my_basis, nbody)

    op_csr = ww[0]*get_csr_3b_op_in_3b_basis(lookup3b,ops[0])
    for i in range(1,num_ops):
        op_csr += ww[i]*get_csr_3b_op_in_3b_basis(lookup3b,ops[i])    
    
    return csr_matrix_tolist_3body(op_csr,lookup3b)