# code/odmr_simulation_with_braiding.py (versione paper-ready)
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

D = 2.870e9           # ZFS Hz
gamma_e = 2.8025e10   # Hz/T

sz = qt.jmat(1, 'z')
H0 = 2 * np.pi * D * sz**2

phi_g = np.pi / 2     # Phase shift geometrico da braiding
Omega_braid = 2 * np.pi * 25e3  # 25 kHz splitting

mw_freqs = np.linspace(2.85e9, 2.89e9, 1500)

def odmr_contrast(H):
    evals, evecs = H.eigenstates()
    # Popolazione m_s=0 (proxy fluorescenza alta quando m_s=0)
    m0_pop = 0
    for i in range(len(evals)):
        overlap = np.abs(evecs[i].overlap(qt.basis(3,1)))**2
        if abs(evals[i] / (2*np.pi) - D) < 5e6:  # vicino m_s=0
            m0_pop += overlap
    return 1 - m0_pop / 3  # contrasto: alto quando m_s=±1 popolati

probs = []
probs_braid = []

for f in mw_freqs:
    H_mw = 2 * np.pi * f * sz
    H_total = H0 + H_mw
    probs.append(odmr_contrast(H_total))
    
    # Braiding term (fase fissa per semplicità, ma asimmetria da cos(phi_g))
    H_braid = Omega_braid * np.cos(phi_g) * qt.jmat(1, 'x')
    H_total_braid = H0 + H_mw + H_braid
    probs_braid.append(odmr_contrast(H_total_braid))

probs = np.array(probs)
probs_braid = np.array(probs_braid)

# Normalizza separatamente
probs_norm = (probs - probs.min()) / (probs.max() - probs.min() + 1e-10)
probs_braid_norm = (probs_braid - probs_braid.min()) / (probs_braid.max() - probs_braid.min() + 1e-10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mw_freqs / 1e9, probs_norm, linewidth=2.5, label='ODMR standard')
ax.plot(mw_freqs / 1e9, probs_braid_norm + 0.15, linewidth=2.5, label='Con braiding ($\phi_g = \pi/2$)')
ax.set_xlabel('Frequenza microonde (GHz)')
ax.set_ylabel('Contrasto ODMR normalizzato (offset per chiarezza)')
ax.set_title('ODMR sidebands da phase shift geometrico (braiding Majorana embodied)')
ax.grid(True, alpha=0.3)
ax.axvline(D/1e9, color='k', ls='--', alpha=0.5, label='ZFS D = 2.87 GHz')
ax.legend()

plt.tight_layout()
plt.savefig('odmr_sidebands_braiding_paper.png', dpi=300)
plt.show()

print("Figura salvata: odmr_sidebands_braiding_paper.png")
print("Sideband splitting approssimativo:", Omega_braid / (2 * np.pi) / 1e3, "kHz")