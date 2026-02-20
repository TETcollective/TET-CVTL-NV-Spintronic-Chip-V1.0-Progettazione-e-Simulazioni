# code/t2_vs_bz_near_gslac.py
# Simulazione T₂ vs B_z vicino GSLAC (clock transition per ingegneria coerenza)

import numpy as np
import matplotlib.pyplot as plt

# Parametri realistici (scalati per T₂ max ~1 ms)
Bz_list = np.linspace(900, 1100, 500)  # G, zoom su GSLAC
B_gslac = 1025  # G

# Decoerenza base (fononi + superficiale)
gamma_other = 800  # Hz (T₂ base ~1.25 ms = 1/gamma_other)

# Decoerenza magnetica: alta lontano dal GSLAC, bassa vicino (clock transition)
gamma_magnetic_base = 5e7  # Hz (sensibilità alta)
sensitivity = np.abs(Bz_list - B_gslac) / 5  # più stretto per picco drammatico
gamma_magnetic = gamma_magnetic_base * sensitivity**2

# Decoerenza totale
gamma_total = gamma_other + gamma_magnetic
T2 = 1 / gamma_total * 1e6  # in μs

plt.figure(figsize=(10, 6))
plt.plot(Bz_list, T2, linewidth=2.5, color='b')
plt.axvline(x=B_gslac, color='r', linestyle='--', linewidth=2, label='GSLAC ~1025 G')
plt.xlabel('Campo B_z (G)')
plt.ylabel('Tempo di coerenza T₂ (μs)')
plt.title('T₂ vs B_z vicino al GSLAC (clock transition)')
plt.annotate('Clock transition: T₂ esteso\nmax ~' + str(round(max(T2))) + ' μs',
             xy=(1025, max(T2)*0.8), xytext=(1050, max(T2)*0.9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('t2_vs_bz_gslac.png', dpi=300)
plt.show()

print("T₂ massimo vicino GSLAC:", round(max(T2)), "μs")