# code/zeeman_gslac_sweep.py
Bz_list = np.linspace(0, 0.15, 200)  # T, fino a ~1500 G
energies = []
for Bz in Bz_list:
    H = H_ZFS + g_e * mu_B * Bz * Sz + H_hyperfine + H_quad
    e = H.eigenenergies() / 1e9
    energies.append(e)

# Plot o analisi gap anticrossing vicino Bz ~0.1025 T