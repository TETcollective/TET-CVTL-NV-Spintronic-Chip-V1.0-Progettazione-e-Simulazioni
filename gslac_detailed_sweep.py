# code/gslac_detailed_sweep.py
# Simulazione dettagliata GSLAC con zoom, livelli distinti e gap anticrossing reale

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Parametri realistici
D = 2.870e9  # Hz, ZFS
g_e = 2.0028
mu_B = 28e9  # GHz/T

# Operatori spin S=1, I=1 per 14N
Sz = qt.jmat(1, 'z')
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Ix = qt.jmat(1, 'x')
Iy = qt.jmat(1, 'y')
Iz = qt.jmat(1, 'z')

# Hamiltonian base
H_ZFS = D * (Sz**2 - 2/3 * qt.qeye(3))

# Iperfine 14N (valori realistici, assiali)
A_par = 2.14e6   # Hz
A_perp = -2.70e6
H_hyperfine = A_par * Sz * Iz + A_perp * (Sx * Ix + Sy * Iy)

# Quadrupolo 14N
P = -4.95e6
H_quad = P * (Iz**2 - 2/3 * qt.qeye(3))

# Sweep zoom su GSLAC (alta risoluzione)
Bz_list = np.linspace(0.101, 0.104, 1200)  # T, 1010-1040 G
energies = []
gaps = []

for Bz in Bz_list:
    H_Zeeman = g_e * mu_B * Bz * Sz
    H = H_ZFS + H_Zeeman + H_hyperfine + H_quad
    e = H.eigenenergies() / 1e9  # in GHz
    energies.append(e)
    
    # Gap anticrossing: differenza tra i due livelli più vicini a 2.87 GHz
    sorted_e = np.sort(e)
    # Trova i due livelli più vicini a D
    idx = np.argsort(np.abs(sorted_e - 2.87))[:2]
    gap = abs(sorted_e[idx[0]] - sorted_e[idx[1]]) * 1e3  # MHz
    gaps.append(gap)

energies = np.array(energies)
gaps = np.array(gaps)

# Plot professionale
fig, ax1 = plt.subplots(figsize=(12, 7))

# Livelli energetici (colori distinti)
ax1.plot(Bz_list * 1e4, energies[:, 0], color='blue', linewidth=1.5, label='Livello più basso')
ax1.plot(Bz_list * 1e4, energies[:, 1], color='green', linewidth=1.5, label='Livello intermedio')
ax1.plot(Bz_list * 1e4, energies[:, 2], color='red', linewidth=1.5, label='Livello più alto')
ax1.axvline(x=1025, color='k', linestyle='--', linewidth=2, label='GSLAC ~1025 G')
ax1.set_xlabel('Campo B_z (G)')
ax1.set_ylabel('Livelli energetici (GHz)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(Bz_list * 1e4, gaps, 'm-', linewidth=2.5, label='Gap anticrossing (MHz)')
ax2.axhline(y=np.min(gaps), color='purple', linestyle=':', label=f'Gap minimo ~{np.min(gaps):.1f} MHz')
ax2.set_ylabel('Gap anticrossing (MHz)')
ax2.legend(loc='upper right')

plt.title('Sweep dettagliato Zeeman e GSLAC con gap mixing')
plt.tight_layout()
plt.savefig('gslac_detailed_plot.png', dpi=300)
plt.show()

print("Gap anticrossing minimo:", np.min(gaps), "MHz a B_z ~", Bz_list[np.argmin(gaps)] * 1e4, "G")