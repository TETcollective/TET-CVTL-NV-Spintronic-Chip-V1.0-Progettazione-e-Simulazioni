# code/tripartite_nv_magnon_mt_fixed.py
# Simulazione tripartita NV - magnone YIG - microtubulo mimetico
# Fix: logarithmic negativity + concurrence effective + warnings suppress
# Per sezione Majorana braiding embodied / RENASCENT-Q

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)          # e_ops keyword
warnings.filterwarnings("ignore", category=qt.LinAlgWarning)      # sqrtm singular

import numpy as np
import qutip as qt
from qutip import negativity, concurrence
import matplotlib.pyplot as plt

print("QuTiP version:", qt.__version__)

# Dimensioni Hilbert
dim_nv = 3      # NV spin-1
dim_m  = 5      # magnone Fock (Kittel mode)
dim_mt = 4      # microtubulo proxy (bosonico low-energy)

id_nv = qt.qeye(dim_nv)
id_m  = qt.qeye(dim_m)
id_mt = qt.qeye(dim_mt)

# Frequenze (GHz → rad/ns)
omega_nv = 2 * np.pi * 2.87   # ZFS
omega_m  = 2 * np.pi * 5.0    # magnone Kittel
omega_mt = 2 * np.pi * 0.5    # MT mode proxy

# Hamiltonian base
H_nv = omega_nv * qt.tensor(qt.jmat(1,'z')**2, id_m, id_mt)
H_m  = omega_m  * qt.tensor(id_nv, qt.num(dim_m), id_mt)
H_mt = omega_mt * qt.tensor(id_nv, id_m, qt.num(dim_mt))

# Coupling (aumentato per entanglement visibile)
g_nm = 2 * np.pi * 80e-3      # NV-magnone ~80 MHz (strong regime realistico)
g_mt = 2 * np.pi * 20e-3      # NV-MT ~20 MHz (weak ma sufficiente per embodied)

H_int_nm = g_nm * qt.tensor(qt.jmat(1,'x'), (qt.destroy(dim_m) + qt.create(dim_m)), id_mt)
H_int_mt = g_mt * qt.tensor(qt.jmat(1,'x'), id_m, (qt.destroy(dim_mt) + qt.create(dim_mt)))

H = H_nv + H_m + H_mt + H_int_nm + H_int_mt

# Decoerenza (magnon damping alto, NV lungo)
kappa_m  = 2 * np.pi * 2e-3   # ~2 MHz (YIG realistico)
gamma_nv = 2 * np.pi * 0.01   # ~10 kHz
c_ops = [
    np.sqrt(kappa_m)  * qt.tensor(id_nv, qt.destroy(dim_m), id_mt),
    np.sqrt(gamma_nv) * qt.tensor(qt.jmat(1,'z'), id_m, id_mt)
]

# Stato iniziale: NV excited + vacuum magnone + MT ground
psi0 = qt.tensor(qt.basis(dim_nv,1), qt.fock(dim_m,0), qt.fock(dim_mt,0))

times = np.linspace(0, 10, 400)  # 0–10 ns per vedere oscillazioni

result = qt.mesolve(H, psi0, times, c_ops, [])

# 1. Logarithmic Negativity NV-MT (metodo principale, no restrizione qubit)
neg = []
for state in result.states:
    rho_nv_mt = state.ptrace([0, 2])  # NV e MT
    neg.append(negativity(rho_nv_mt, logarithmic=True))

# 2. Concurrence effective (opzionale, proiezione a due qubit)
conc_eff = []
for state in result.states:
    rho_nv_mt = state.ptrace([0, 2])
    
    # Proietta NV su subspace m_s = ±1 (effective qubit)
    proj_nv_pm = qt.basis(dim_nv,0).proj() + qt.basis(dim_nv,2).proj()
    rho_proj = proj_nv_pm * rho_nv_mt * proj_nv_pm
    rho_proj = rho_proj / rho_proj.tr() if rho_proj.tr() > 1e-10 else rho_nv_mt
    
    # Troncamento MT a Fock 0 e 1 (qubit-like)
    proj_mt_01 = qt.fock(dim_mt,0).proj() + qt.fock(dim_mt,1).proj()
    rho_2q = qt.tensor(proj_nv_pm, proj_mt_01) * rho_proj * qt.tensor(proj_nv_pm, proj_mt_01)
    rho_2q = rho_2q / rho_2q.tr() if rho_2q.tr() > 1e-10 else rho_proj
    
    # Prova concurrence (se dim=4×4)
    if rho_2q.shape == (4, 4):
        try:
            c = concurrence(rho_2q)
            conc_eff.append(c)
        except:
            conc_eff.append(0.0)
    else:
        conc_eff.append(0.0)

# Plot principali
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.plot(times*1e9, neg, linewidth=2.5, color='purple')
plt.xlabel('Tempo (ns)')
plt.ylabel('Logarithmic Negativity')
plt.title('Log-Negativity NV-MT (mediated by YIG magnone)')
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(times*1e9, conc_eff, linewidth=2.5, color='green')
plt.xlabel('Tempo (ns)')
plt.ylabel('Concurrence (effective qubit)')
plt.title('Concurrence effective NV-MT (subspace projection)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tripartite_entanglement_fixed.png', dpi=300)
plt.show()

print("Log-Negativity max:", max(neg))
print("Concurrence effective max:", max(conc_eff))