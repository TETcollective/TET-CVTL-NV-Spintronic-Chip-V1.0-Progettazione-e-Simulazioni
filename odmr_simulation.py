# code/odmr_simulation.py
"""
Simulazione curva ODMR per centro NV^- (ground state)
- ZFS a 2.87 GHz
- Optional: strain shift su D
- Optional: hyperfine splitting semplice (per 14N o 15N)
- Linea Lorentziana con broadening
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Parametri fisici
D0 = 2.870e9          # Hz, zero-field splitting
gamma = 5e6           # Hz, FWHM Lorentziana (tipica in bulk/shallow)
A_contrast = 0.25     # contrasto fluorescenza (25%)
freq_range = 0.1e9    # range intorno a D0 (100 MHz)
n_points = 1000

# Frequenze microonde da simulare
nu = np.linspace(D0 - freq_range/2, D0 + freq_range/2, n_points)

# Funzione Lorentziana singola
def lorentzian(nu, nu0, gamma, A):
    return -A * (gamma/2 / np.pi) / ((nu - nu0)**2 + (gamma/2)**2)

# Caso base: doppio dip degenerato m_s = ±1
delta_f = 0.0         # shift da strain (in Hz)
odmr_base = lorentzian(nu, D0 + delta_f, gamma, A_contrast) + \
            lorentzian(nu, D0 + delta_f, gamma, A_contrast)

# Caso con strain shift (es. SAW modulation)
strain_shift = 10e6   # es. +10 MHz shift da strain
odmr_strain = lorentzian(nu, D0 + strain_shift, gamma, A_contrast) + \
              lorentzian(nu, D0 + strain_shift, gamma, A_contrast)

# Caso semplice con hyperfine 14N (tripletto approssimato)
A_hf_14N = 2.5e6      # splitting medio ~2.5 MHz
odmr_hyperfine = lorentzian(nu, D0 - A_hf_14N, gamma, A_contrast*0.7) + \
                 lorentzian(nu, D0, gamma, A_contrast) + \
                 lorentzian(nu, D0 + A_hf_14N, gamma, A_contrast*0.7)

# Plot
plt.figure(figsize=(10, 6))
plt.plot((nu - D0)/1e6, odmr_base / A_contrast, label='Base (B=0, ε=0)', linewidth=2)
plt.plot((nu - D0)/1e6, odmr_strain / A_contrast, label=f'Strain shift +{strain_shift/1e6:.1f} MHz', linestyle='--')
plt.plot((nu - D0)/1e6, odmr_hyperfine / A_contrast, label='Con hyperfine ¹⁴N (approx)', linestyle=':')

plt.xlabel('Detuning dalla risonanza centrale (MHz)')
plt.ylabel('Fluorescenza normalizzata (contrasto)')
plt.title('Simulazione curva ODMR centro NV$^-$ (QuTiP-inspired)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Opzionale: salva come immagine per il paper
# plt.savefig('odmr_simulation.png', dpi=300)
print("ODMR simulata: doppio dip base centrato a 2.87 GHz")