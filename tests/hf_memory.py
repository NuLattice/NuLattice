def estimate_all_hf_memory(L, num_particles=16):
    """
    Revised estimation based on Memray diagnostic results for L=6,7,8.
    Reflects the impact of the eigh workspace and buffer counts.
    """
    n_stat = (L**3) * 4
    f64_size = 8
    i64_size = 8
    
    # Interaction scaling (SoA format)
    # n_v2 (27 neighbors) and n_v3 (approx 100 neighbors)
    n_v2 = n_stat * 27
    n_v3 = n_stat * 100

    def calc_soa_int():
        # Indices (i64) + Values (f64)
        v2 = n_v2 * (5 * i64_size + 1 * f64_size) / (1024**2)
        v3 = n_v3 * (7 * i64_size + 1 * f64_size) / (1024**2)
        return v2 + v3

    # --- AOES (Legacy) Logic ---
    # 6 dense buffers + the ~5x Workspace used by eigh
    # Plus punishing Python list-of-list overhead
    aos_dense_count = 6 + 5.2 
    aos_dense = (n_stat**2) * f64_size * aos_dense_count / (1024**2)
    aos_int = (n_v2 * 450 + n_v3 * 800) / (1024**2)
    aos_total = aos_dense + aos_int

    # --- SoA (Current Pass-by-Ref + Dense EIGH) ---
    # This reflects first memray result (e.g., L=8 was ~389 MiB)
    # EIGH is dominant memory consumer
    soa_dense_count = 6 + 5.1 
    soa_dense = (n_stat**2) * f64_size * soa_dense_count / (1024**2)
    soa_int = calc_soa_int()
    soa_total = soa_dense + soa_int

    # --- Subspace (Current Optimized with EIGSH) ---
    # This reflects latest memray result (e.g., L=8 was ~259 MiB)
    # Largest allocations now initial allocation of memory
    sub_dense_count = 6 
    sub_dense = (n_stat**2) * f64_size * sub_dense_count / (1024**2)
    sub_int = calc_soa_int()
    sub_total = sub_dense + sub_int

    # --- Matrix-Free (Future Target) ---
    # Only 1 dense buffer (Density) + Orbitals (N*A)
    # No H or Gamma matrices stored.
    mf_dense = ((n_stat**2) * f64_size * 1 + (n_stat * num_particles) * f64_size * 2) / (1024**2)
    mf_int = calc_soa_int()
    mf_total = mf_dense + mf_int

    # Printing Results
    header = f"--- HF Memory Comparison: L={L} (N={n_stat}, A={num_particles}) ---"
    template = "{:<15} | {:>15} | {:>15} | {:>15} | {:>15}"
    row_str = template.format("Component", "AoS (Legacy)", "SoA (Dense)", "SoA (Subspace)", "Matrix-Free")
    print("-" * len(row_str))
    print(header)
    print("-" * len(row_str))
    print(row_str)
    print("-" * len(row_str))
    print(template.format("Dense Buffers", f"{aos_dense:.1f} MB", f"{soa_dense:.1f} MB", f"{sub_dense:.1f} MB", f"{mf_dense:.1f} MB"))
    print(template.format("Interactions", f"{aos_int:.1f} MB", f"{soa_int:.1f} MB", f"{sub_int:.1f} MB", f"{mf_int:.1f} MB"))
    print("-" * len(row_str))
    print(template.format("TOTAL", f"{aos_total:.2f} MB", f"{soa_total:.2f} MB", f"{sub_total:.2f} MB", f"{mf_total:.2f} MB"))
    print("-" * len(row_str))
    print("\n")

if __name__ == "__main__":
    for L in [6, 7, 8, 16]:
        estimate_all_hf_memory(L)
