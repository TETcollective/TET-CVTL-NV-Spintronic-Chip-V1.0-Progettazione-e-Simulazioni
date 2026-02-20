# code/nv_hyperfine_strain.py
# Hamiltonian iperfine completo NV con strain + plot livelli energetici
# Per sezione "Struttura Iperfine e Quadrupolo (14N vs 15N)"

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

D0 = 2.870e9
dD_de = -14.6e9
gamma_e = 2.8025e10
gamma_n_14 = 4.315e6
gamma_n_15 = -6.076e6

A_par_14 = 81e6
A_perp_14 = -52e6
Qzz_14 = -4.9e6

A_par_15 = -114e6
A_perp_15 = 73e6

B = 0.1025  # T

epsilon_vals = np.linspace(-0.1, 0.1, 200)

def nv_hyperfine_ham(epsilon=0.0, isotope='14N', Bz=B):
    D = D0 + dD_de * epsilon
    
    Hzfs = D * qt.jmat(1, 'z')**2
    Hze = gamma_e * Bz * qt.jmat(1, 'z')
    
    if isotope == '14N':
        A = qt.Qobj(np.diag([A_par_14, A_perp_14, A_perp_14]))
        Q = qt.Qobj([[Qzz_14, 0, 0], [0, -Qzz_14/2, 0], [0, 0, -Qzz_14/2]])
        Hhf = qt.tensor(qt.jmat(1, 'z'), A[0,0] * qt.sigmaz()) + \
              qt.tensor(qt.jmat(1, 'x'), A[1,1] * qt.sigmax()) + \
              qt.tensor(qt.jmat(1, 'y'), A[2,2] * qt.sigmay())
        Hq = qt.tensor(qt.qeye(3), Q)
        Hnze = -gamma_n_14 * Bz * qt.sigmaz()
        H_total = Hzfs + Hze + Hhf + Hq + Hnze
        dims = [3, 3]
    else:
        A = qt.Qobj(np.diag([A_par_15, A_perp_15, A_perp_15]))
        Hhf = qt.tensor(qt.jmat(1, 'z'), A[0,0] * qt.sigmaz()) + \
              qt.tensor(qt.jmat(1, 'x'), A[1,1] * qt.sigmax()) + \
              qt.tensor(qt.jmat(1, 'y'), A[2,2] * qt.sigmay())
        Hnze = -gamma_n_15 * Bz * qt.sigmaz()
        H_total = Hzfs + Hze + Hhf + Hnze
        dims = [3, 2]
    
    return H_total, dims

energies_14 = []
energies_15 = []

for eps in epsilon_vals:
    H14, _ = nv_hyperfine_ham(eps, '14N', B)
    ev14 = H14.eigenenergies() / 1e9
    energies_14.append(ev14)
    
    H15, _ = nv_hyperfine_ham(eps, '15N', B)
    ev15 = H15.eigenenergies() / 1e9
    energies_15.append(ev15)

energies_14 = np.array(energies_14)
energies_15 = np.array(energies_15)

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 14N
for i in range(energies_14.shape[1]):
    axs[0].plot(epsilon_vals * 100, energies_14[:, i], linewidth=1.8)
axs[0].set_xlabel('Strain relativo (%)')
axs[0].set_ylabel('Energia (GHz)')
axs[0].set_title(r'$^{14}$N (I=1): iperfine + quadrupolo + strain')
axs[0].grid(True, alpha=0.3)
axs[0].axhline(D0/1e9, color='k', ls='--', alpha=0.6, label=r'$D_0$ = 2.87 GHz')
axs[0].legend()

# 15N
for i in range(energies_15.shape[1]):
    axs[1].plot(epsilon_vals * 100, energies_15[:, i], linewidth=1.8)
axs[1].set_xlabel('Strain relativo (%)')
axs[1].set_title(r'$^{15}$N (I=1/2): iperfine + strain')
axs[1].grid(True, alpha=0.3)
axs[1].axhline(D0/1e9, color='k', ls='--', alpha=0.6, label=r'$D_0$ = 2.87 GHz')
axs[1].legend()

fig.suptitle('Shift livelli energetici NV vs strain SAW (B = 0.1025 T)')
fig.tight_layout()
plt.savefig('nv_hyperfine_strain_levels.png', dpi=300)
plt.show()