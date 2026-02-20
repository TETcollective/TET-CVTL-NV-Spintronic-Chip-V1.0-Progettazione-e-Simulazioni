"""
Heatmap Ottimizzazione Retrocausale Aurea XX+ZZ per RENASCENT-Q
===============================================================
Analisi parametrica bidimensionale (k vs alpha) con e senza scaling aureo β = φ⁻² sul termine ZZ.
Genera heatmap_xxzz_beta.pdf con confronto full-strength vs weak retrocausal bias.

Obiettivo: mostrare che scaling aureo permette C equivalente con coupling ZZ ridotto ~62%.



import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

phi = (1 + np.sqrt(5)) / 2
beta = 1 / phi**2  # ≈ 0.381966

def compute_concurrence_final(
    k, alpha, gamma_relax=0.012, t_braid=10.0,
    kick_interval=4.0, kick_duration=0.5, use_beta=True
):
    gamma_deph = gamma_relax * 0.6
    k_xx = k
    k_zz = alpha * k * (beta if use_beta else 1.0)

    psi0 = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) +
            qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()

    proj11 = qt.basis(2,1).proj()
    H_phase = (np.pi/3) * qt.tensor(proj11, proj11)
    swap_like = qt.Qobj([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]], dims=[[2,2],[2,2]])
    H_main = H_phase + 0.8 * swap_like

    sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))
    sm2 = qt.tensor(qt.qeye(2), qt.sigmam())
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

    c_ops = [np.sqrt(gamma_relax)*sm1, np.sqrt(gamma_relax)*sm2,
             np.sqrt(gamma_deph/2)*sz1, np.sqrt(gamma_deph/2)*sz2]

    opts = {'store_states': True, 'nsteps': 12000, 'rtol': 5e-8, 'atol': 1e-10}

    H_kick = k_xx * qt.tensor(qt.sigmax(), qt.sigmax()) + \
             k_zz * qt.tensor(qt.sigmaz(), qt.sigmaz())

    current = psi0
    t_now = 0.0

    for ks in np.arange(kick_interval, t_braid + 1e-6, kick_interval):
        if ks - t_now > 0.02:
            t_free = np.linspace(t_now, ks, int((ks - t_now)*80) + 2)
            res = qt.mesolve(H_main, current, t_free, c_ops=c_ops, options=opts)
            current = res.states[-1]
            t_now = ks

        t_end_k = min(ks + kick_duration, t_braid)
        if t_end_k > ks + 1e-5:
            t_k = np.linspace(ks, t_end_k, int(kick_duration*150) + 2)
            res_k = qt.mesolve(H_kick, current, t_k, c_ops=c_ops, options=opts)
            current = res_k.states[-1]
            t_now = t_end_k

    if t_braid - t_now > 0.02:
        t_last = np.linspace(t_now, t_braid, int((t_braid - t_now)*80) + 2)
        res_last = qt.mesolve(H_main, current, t_last, c_ops=c_ops, options=opts)
        current = res_last.states[-1]

    rho = current * current.dag() if current.isket else current
    return qt.concurrence(rho)


# Griglia fine intorno al picco ottimale
g_values = np.linspace(9.0, 13.0, 25)
alpha_values = np.linspace(0.05, 0.20, 16)

C_beta = np.zeros((len(alpha_values), len(g_values)))
C_nobeta = np.zeros_like(C_beta)

print("Calcolo heatmap con β (retrocausale aureo)...")
for i, alpha in enumerate(alpha_values):
    for j, g in enumerate(g_values):
        C_beta[i,j] = compute_concurrence_final(g, alpha, use_beta=True)
        C_nobeta[i,j] = compute_concurrence_final(g, alpha, use_beta=False)

# Plot affiancato
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

im1 = ax1.imshow(C_beta, origin='lower', aspect='auto', cmap='magma',
                 extent=[g_values.min(), g_values.max(), alpha_values.min(), alpha_values.max()],
                 norm=Normalize(vmin=0.65, vmax=0.76))
ax1.set_title(r'Con $\beta$ su ZZ (weak retrocausal bias)')
ax1.set_xlabel(r'$k$ nominale (XX full, ZZ $\times\beta$)')
ax1.set_ylabel(r'$\alpha$ (prima di $\beta$)')
fig.colorbar(im1, ax=ax1, label=r'Concurrence finale ($t=10$)')

im2 = ax2.imshow(C_nobeta, origin='lower', aspect='auto', cmap='magma',
                 extent=[g_values.min(), g_values.max(), alpha_values.min(), alpha_values.max()],
                 norm=Normalize(vmin=0.65, vmax=0.76))
ax2.set_title('Senza $\beta$ (full strength)')
ax2.set_xlabel(r'$k$ nominale')
fig.colorbar(im2, ax=ax2, label=r'Concurrence finale ($t=10$)')

plt.suptitle(r'Ottimizzazione kicks XX+ZZ – $\gamma_\mathrm{relax}=0.012$')
plt.tight_layout()
plt.savefig('heatmap_xxzz_beta.pdf', dpi=400, bbox_inches='tight')
plt.show()

print(f"Massimo con β: {np.max(C_beta):.4f}")
print(f"Massimo senza β: {np.max(C_nobeta):.4f}")