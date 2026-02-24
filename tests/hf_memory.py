def estimate_all_hf_memory(L, num_particles=16):
    """
    Prints a comparison of memory usage for AoS, SoA, and Optimized 
    subspace implementations for a given lattice size L.
    """
    # N = L^3 * 4 (spin/isospin)
    n_stat = (L**3) * 4
    f64_size = 8
    i64_size = 8
    
    # Interaction heuristics
    n_v2 = n_stat * 27
    n_v3 = n_stat * 100

    def calc_soa_int():
        v2 = n_v2 * 5 * i64_size / (1024**2)
        v3 = n_v3 * 7 * i64_size / (1024**2)
        return v2 + v3

    # AoS Logic
    # 6 dense N*N buffers + Python Object Overhead for interactions
    aos_dense = (n_stat**2) * f64_size * 6 / (1024**2)
    aos_int = (n_v2 * 250 + n_v3 * 400) / (1024**2)
    aos_total = aos_dense + aos_int

    # SoA Logic
    # 6 dense N*N buffers + Contiguous Array interactions
    soa_dense = (n_stat**2) * f64_size * 6 / (1024**2)
    soa_int = calc_soa_int()
    soa_total = soa_dense + soa_int

    # Optimized Logic
    # 2 dense N*N (H, Gamma) + 4 subspace N*A (Wavefunctions)
    opt_dense_mat = (n_stat**2) * f64_size * 2 / (1024**2)
    opt_subspace_mat = (n_stat * num_particles) * f64_size * 4 / (1024**2)
    opt_dense = opt_dense_mat + opt_subspace_mat
    opt_int = calc_soa_int() # Uses SoA format
    opt_total = opt_dense + opt_int

    # Printing Results
    header = f"--- HF Memory Comparison: L={L} (N={n_stat}, A={num_particles}) ---"
    template = "{:<15} | {:>15} | {:>15} | {:>15}"
    lines = len(template.format("Component", "AoS (Legacy)", "SoA (Current)", "Optimized"))
    print("-" * lines)
    print(header)
    print("-" * lines)
    
    print(template.format("Component", "AoS (Legacy)", "SoA (Current)", "Optimized"))
    print("-" * lines)
    print(template.format("Dense Buffers", f"{aos_dense:.2f} MB", f"{soa_dense:.2f} MB", f"{opt_dense:.2f} MB"))
    print(template.format("Interactions", f"{aos_int:.2f} MB", f"{soa_int:.2f} MB", f"{opt_int:.2f} MB"))
    print("-" * lines)
    print(template.format("TOTAL", f"{aos_total:.2f} MB", f"{soa_total:.2f} MB", f"{opt_total:.2f} MB"))
    print("-" * lines)
    print("\n")

if __name__ == "__main__":
    for L in range(20):
        estimate_all_hf_memory(L)
