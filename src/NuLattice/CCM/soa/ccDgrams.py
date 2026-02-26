import torch


def pAB(val):
    """
    Permutator for ab indices (0, 1).
    Returns val^{ab}_{ij} - val^{ba}_{ij}
    """
    return val - val.permute(1, 0, 2, 3)


def pIJ(val):
    """
    Permutator for ij indices (2, 3).
    Returns val^{ab}_{ij} - val^{ab}_{ji}
    """
    return val - val.permute(0, 1, 3, 2)


def v_ppph_dgrams(v_ppph_soa, t1, t2):
    indices, values = v_ppph_soa
    pnum, hnum = t1.shape
    device = t1.device
    dtype = t1.dtype

    ret0 = torch.zeros((pnum, hnum), device=device, dtype=dtype)
    ret1 = torch.zeros((pnum, pnum), device=device, dtype=dtype)
    ret2 = torch.zeros((pnum, pnum, hnum, hnum), device=device, dtype=dtype)
    ret3 = torch.zeros((pnum, pnum), device=device, dtype=dtype)
    ret4 = torch.zeros((pnum, pnum, hnum, hnum), device=device, dtype=dtype)
    ret5 = torch.zeros((pnum, hnum, hnum, hnum), device=device, dtype=dtype)
    ret6 = torch.zeros((pnum, hnum, hnum, hnum), device=device, dtype=dtype)

    if values.numel() == 0:
        return ret0, ret1, ret2, ret3, ret4, ret5, ret6

    # V indices: c, d, a, k (0, 1, 2, 3)
    idx_c, idx_d, idx_a, idx_k = indices[0], indices[1], indices[2], indices[3]
    nnz = values.shape[0]
    h_range = torch.arange(hnum, device=device)

    # Diagram 0: T1 contribution
    # ret0[a, i] -= 0.5 * sum_{cdk} V[c,d,a,k] * T2[c,d,k,i]
    # index_add on dim 0 (a) with the full (nnz, hnum) block
    term_0 = -0.5 * values.unsqueeze(1) * t2[idx_c, idx_d, idx_k, :]
    ret0.index_add_(0, idx_a, term_0)

    # Diagrams 1 & 3: X_pp contribution (Simple scatter)
    term_1_3 = values * t1[idx_c, idx_k]
    ret1.index_put_((idx_a, idx_d), -term_1_3, accumulate=True)
    ret3.index_put_((idx_d, idx_a), term_1_3, accumulate=True)

    s2 = torch.tensor(ret2.stride(), device=device)
    s5 = torch.tensor(ret5.stride(), device=device)

    # pre-broadcast indices for 2D expansion (1 hole index j)
    c_2d = idx_c.view(-1, 1).expand(nnz, hnum)
    d_2d = idx_d.view(-1, 1).expand(nnz, hnum)
    a_2d = idx_a.view(-1, 1).expand(nnz, hnum)
    k_2d = idx_k.view(-1, 1).expand(nnz, hnum)
    j_2d = h_range.view(1, -1).expand(nnz, hnum)

    # Diagram 2: ret2[c, d, j, k] += V[c,d,a,k] * T1[a, j]
    term_2 = values.view(-1, 1) * t1[idx_a, :]
    flat_idx_2 = c_2d * s2[0] + d_2d * s2[1] + j_2d * s2[2] + k_2d * s2[3]
    ret2.view(-1).index_add_(0, flat_idx_2.reshape(-1), term_2.reshape(-1))

    # Diagram 4: ret4[a, d, j, k] += V[c,d,a,k] * T1[c, j]
    term_4 = values.view(-1, 1) * t1[idx_c, :]
    flat_idx_4 = a_2d * s2[0] + d_2d * s2[1] + j_2d * s2[2] + k_2d * s2[3]
    ret4.view(-1).index_add_(0, flat_idx_4.reshape(-1), term_4.reshape(-1))

    # Pre-broadcast indices for 3D expansion (2 hole indices i, j)
    a_3d = idx_a.view(-1, 1, 1).expand(nnz, hnum, hnum)
    k_3d = idx_k.view(-1, 1, 1).expand(nnz, hnum, hnum)
    i_3d = h_range.view(1, -1, 1).expand(nnz, hnum, hnum)
    j_3d = h_range.view(1, 1, -1).expand(nnz, hnum, hnum)

    # Diagram 5: ret5[a, i, j, k] += V[c,d,a,k] * T2[c,d,i,j]
    term_5 = values.view(-1, 1, 1) * t2[idx_c, idx_d, :, :]
    flat_idx_5 = a_3d * s5[0] + i_3d * s5[1] + j_3d * s5[2] + k_3d * s5[3]
    ret5.view(-1).index_add_(0, flat_idx_5.reshape(-1), term_5.reshape(-1))

    # Diagram 6: ret6[a, i, j, k] += V[c,d,a,k] * (T1[c,i]*T1[d,j])
    # We compute (T1*T1) intermediate cdij only inside this block to save memory
    doubleT1 = torch.einsum("ci, dj -> cdij", t1, t1)
    term_6 = values.view(-1, 1, 1) * doubleT1[idx_c, idx_d, :, :]
    ret6.view(-1).index_add_(0, flat_idx_5.reshape(-1), term_6.reshape(-1))

    return ret0, ret1, ret2, ret3, ret4, ret5, ret6


def v_pppp_dgrams(v_pppp_soa, t1, t2):
    """
    Calculates both diagrams using v_pppp (Sparse SoA).

    :param v_pppp_soa: Tuple (indices, values).
                       indices shape (4, N_nnz) corresponding to [a, b, c, d]
                       values shape (N_nnz)
    """
    indices, values = v_pppp_soa
    idx_a, idx_b, idx_c, idx_d = indices[0], indices[1], indices[2], indices[3]

    pnum, hnum = t1.shape
    device = t1.device
    dtype = t1.dtype

    ret1 = torch.zeros((pnum, pnum, hnum, hnum), device=device, dtype=dtype)
    ret2 = torch.zeros((pnum, pnum, hnum, hnum), device=device, dtype=dtype)

    doubleT1 = torch.einsum("ci, dj -> cdij", t1, t1)

    # ret1[a, b, :, :] += v * t2[c, d, :, :]
    term_1 = values.view(-1, 1, 1) * t2[idx_c, idx_d, :, :]
    ret1.index_put_((idx_a, idx_b), term_1, accumulate=True)

    # ret2[a, b, :, :] += v * doubleT1[c, d, :, :]
    term_2 = values.view(-1, 1, 1) * doubleT1[idx_c, idx_d, :, :]
    ret2.index_put_((idx_a, idx_b), term_2, accumulate=True)

    return ret1, ret2


def dgram_akci_ck(v_phph, t1):
    return -torch.einsum("akci, ck -> ai", v_phph, t1)


def dgram_ck_acik(f_ph, t2):
    return torch.einsum("ck, acik -> ai", f_ph, t2)


def dgram_cikl_cakl(v_phhh, t2):
    return -0.5 * torch.einsum("cikl, cakl -> ai", v_phhh, t2)


def dgram_cdkl_ck_dali(v_pphh, t1, t2):
    # NOTE(vivek): Original used greedy optimization.
    # standard Torch einsum is usually sufficient, but we could decompose like below
    # cdkl, ck, dali -> ai
    # temp[d,l,i] = sum_k,c (v[c,d,k,l] * t1[c,k]) -> Not quite, check indices carefully
    # v[cdkl] * t1[ck] -> temp[dli] -> temp[dli] * t2[dali] -> result
    return torch.einsum("cdkl, ck, dali -> ai", v_pphh, t1, t2)


def dgram_ck_ci(f_ph, t1):
    return -0.5 * torch.einsum("ck, ci -> ki", f_ph, t1)


def dgram_ck_ak(f_ph, t1):
    return -0.5 * torch.einsum("ck, ak -> ac", f_ph, t1)


def dgram_bijk_bj(v_phhh, t1):
    return -torch.einsum("bijk, bj -> ki", v_phhh, t1)


def dgram_cdlk_cdli(v_pphh, t2):
    return -0.5 * torch.einsum("cdlk, cdli -> ki", v_pphh, t2)


def dgram_dckl_dakl(v_pphh, t2):
    return -0.5 * torch.einsum("dckl, dakl -> ac", v_pphh, t2)


def dgram_cdlk_cl_di(v_pphh, t1):
    # 'cdlk, cl, di -> ki'
    # Decomposable: (v_pphh * t1_cl) * t1_di
    return -0.5 * torch.einsum("cdlk, cl, di -> ki", v_pphh, t1, t1)


def dgram_cdkl_dk_al(v_pphh, t1):
    return 0.5 * torch.einsum("cdkl, dk, al -> ac", v_pphh, t1, t1)


def dgram_klij_abkl(v_hhhh, t2):
    return 0.5 * torch.einsum("klij, abkl -> abij", v_hhhh, t2)


def dgram_bkcj_acik(v_phph, t2):
    # -P(ij)P(ab)V^{bk}_{cj}t^{cb}_{ik}
    contracted = torch.einsum("bkcj, acik -> abij", v_phph, t2)
    return -pIJ(pAB(contracted))


def dgram_bkij_ak(v_phhh, t1):
    contracted = torch.einsum("bkij, ak -> abij", v_phhh, t1)
    return pAB(contracted)


def dgram_cdkl_acik_dblj(v_pphh, t2):
    # 'cdkl, acik, dblj -> abij'
    # This is an N^6 contraction.
    return 0.5 * pIJ(pAB(torch.einsum("cdkl, acik, dblj -> abij", v_pphh, t2, t2)))


def dgram_cdkl_cdij_abkl(v_pphh, t2):
    return 0.25 * torch.einsum("cdkl, cdij, abkl -> abij", v_pphh, t2, t2)


def dgram_klij_ak_bl(v_hhhh, t1):
    return 0.5 * pAB(torch.einsum("klij, ak, bl -> abij", v_hhhh, t1, t1))


def dgram_bkci_ak_cj(v_phph, t1):
    return -pAB(pIJ(torch.einsum("bkci, ak, cj -> abij", v_phph, t1, t1)))


def dgram_cikl_ck_ablj(v_phhh, t1, t2):
    return -pIJ(torch.einsum("cikl, ck, ablj -> abij", v_phhh, t1, t2))


def dgram_da_dbij(v_ppph_res, t2):
    # v_ppph_res is dense here (it's an intermediate result X_{ad})
    contracted = torch.einsum("da, dbij -> abij", v_ppph_res, t2)
    return -pAB(contracted)


def dgram_acik_bcjk(v_ppph_res, t2):
    # v_ppph_res is dense X_{acij}
    contracted = torch.einsum("acik, bcjk -> abij", v_ppph_res, t2)
    return pIJ(pAB(contracted))


def dgram_cikl_al_bcjk(v_phhh, t1, t2):
    return -pIJ(pAB(torch.einsum("cikl, al, bcjk -> abij", v_phhh, t1, t2)))


def dgram_cjkl_ci_abkl(v_phhh, t1, t2):
    return 0.5 * pIJ(torch.einsum("cjkl, ci, abkl -> abij", v_phhh, t1, t2))


def dgram_bijk_ak1(v_ppph_res, t1):
    # v_ppph_res is dense X_{bijk}
    return 0.5 * pAB(torch.einsum("bijk, ak -> abij", v_ppph_res, t1))


def dgram_bijk_ak2(v_ppph_res, t1):
    return 0.5 * pIJ(pAB(torch.einsum("bijk, ak -> abij", v_ppph_res, t1)))


def dgram_cjkl_ci_ak_bl(v_phhh, t1):
    return 0.5 * pIJ(pAB(torch.einsum("cjkl, ci, ak, bl -> abij", v_phhh, t1, t1, t1)))


def dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2):
    return 0.25 * pIJ(torch.einsum("cdkl, ci, dj, abkl -> abij", v_pphh, t1, t1, t2))


def dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2):
    return 0.25 * pAB(torch.einsum("cdkl, ak, bl, cdij -> abij", v_pphh, t1, t1, t2))


def dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2):
    return pIJ(pAB(torch.einsum("cdkl, ci, bl, adkj -> abij", v_pphh, t1, t1, t2)))


def dgram_cdkl_ci_ak_dj_bl(v_pphh, t1):
    return 0.25 * pIJ(
        pAB(torch.einsum("cdkl, ci, ak, dj, bl -> abij", v_pphh, t1, t1, t1, t1))
    )


def dgram_cdkl_bdkl(v_pphh, t2):
    return -0.5 * torch.einsum("cdkl, bdkl -> bc", v_pphh, t2)


def dgram_cdkl_cdjl(v_pphh, t2):
    return -0.5 * torch.einsum("cdkl, cdjl -> kj", v_pphh, t2)


def dgram_ck_bk(f_ph, t1):
    return -torch.einsum("ck, bk -> bc", f_ph, t1)


def dgram_ck_cj(f_ph, t1):
    return -torch.einsum("ck, cj -> kj", f_ph, t1)


def dgram_cdlk_cl_dj(v_pphh, t1):
    return -torch.einsum("cdlk, cl, dj -> kj", v_pphh, t1, t1)


def dgram_cdlk_dk_bl(v_pphh, t1):
    return -torch.einsum("cdlk, dk, bl -> bc", v_pphh, t1, t1)
