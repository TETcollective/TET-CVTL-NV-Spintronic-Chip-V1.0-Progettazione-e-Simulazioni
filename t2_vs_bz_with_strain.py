# code/t2_vs_bz_with_strain.py
# T₂ vs B_z con shift GSLAC dovuto a strain SAW (versione realistica)

import numpy as np
import matplotlib.pyplot as plt

Bz_list = np.linspace(900, 1100, 800)  # G, risoluzione alta
B_gslac_base = 1025  # G base

gamma_other = 400  # Hz (T₂ base ~2.5 ms con multilayer)
gamma_magnetic_base = 1e8  # Hz (sensibilità alta)

strain_levels = [0.0, 0.05, 0.10]  # % strain relativo
colors = ['b', 'g', 'r']
labels = [f'Strain {s*100:.1f}%' for s in strain_levels]

plt.figure(figsize=(12, 7))

for strain, color, lbl in zip(strain_levels, colors, labels):
    shift_D = -14.6e9 * strain  # Hz, shift negativo tipico
    B_gslac_shifted = B_gslac_base + shift_D / (2.0028 * 28e9 * 1e4)  # G shift
    sensitivity = np.abs(Bz_list - B_gslac_shifted) / 3  # più stretto
    gamma_magnetic = gamma_magnetic_base * sensitivity**2
    gamma_total = gamma_other + gamma_magnetic
    T2 = 1 / gamma_total * 1e6  # μs
    plt.plot(Bz_list, T2, linewidth=2.5, color=color, label=lbl)

plt.axvline(x=B_gslac_base, color='k', linestyle='--', linewidth=2, label='GSLAC base ~1025 G')
plt.xlabel('Campo B_z (G)')
plt.ylabel('Tempo di coerenza T₂ (μs)')
plt.title('T₂ vs B_z con shift GSLAC da strain SAW')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('t2_vs_bz_with_strain.png', dpi=300)
plt.show()

print("T₂ massimo varia con strain: picco spostato con GSLAC.")