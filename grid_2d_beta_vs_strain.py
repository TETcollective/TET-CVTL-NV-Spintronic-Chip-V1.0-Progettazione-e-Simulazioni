"""
Grid Search 2D: β vs strain_amp_base per RENASCENT-Q
====================================================
Calcola negentropy von Neumann e concurrence media multi-bipartizione
su griglia 2D β × strain_amp_base (N=6 qubit GHZ-like).
Genera due heatmap: negentropy e concurrence media.

Output: heatmap_negentropy_beta_strain.jpg
        heatmap_concurrence_beta_strain.jpg




import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def compute_negentropy(rho):
    """Negentropy = -S_vN"""
    if rho.type == 'ket':
        rho = rho * rho.dag()
    evals = rho.eigenenergies()
    evals = evals[evals > 1e-14]
    if len(evals) == 0:
        return 0.0
    S = -np.sum(evals * np.log(evals + 1e-20))
    return -S


def average_concurrence(rho, N, n_bipart=10):
    """Concurrence media su n_bipart bipartizioni casuali"""
    if N < 2:
        return 0.0
    concs = []
    for _ in range(n_bipart):
        perm = np.random.permutation(N)
        cut = np.random.randint(1, N)
        A = perm[:cut]
        B = perm[cut:]
        if len(A) == 0 or len(B) == 0:
            continue
        rho_ab = rho.ptrace([A[0] % N, B[0] % N])  # Fix: Pass indices as a list
        C = qt.concurrence(rho_ab)
        if not np.isnan(C):
            concs.append(C)
    return np.mean(concs) if concs else 0.0


def grid_search_2d_beta_strain(
    beta_list=np.linspace(0.30, 0.45, 16),
    strain_list=np.linspace(0.05, 0.13, 9),
    N=6,
    epsilon=0.0045,
    g0=0.17,
    T1=110.0,
    T2=38.0,
    t_weak=0.9,
    omega_saw=2 * np.pi * 5.5,
    n_steps=180,
    n_bipart=8
):
    gamma1 = 1.0 / T1
    gamma_phi = max(0, 1.0 / T2 - gamma1 / 2.0)

    Sz_total = sum(qt.tensor([qt.qeye(2)]*i + [qt.sigmaz()] + [qt.qeye(2)]*(N-1-i)) for i in range(N))
    Sx_total = sum(qt.tensor([qt.qeye(2)]*i + [qt.sigmax()] + [qt.qeye(2)]*(N-1-i)) for i in range(N))

    ghz_up   = qt.tensor([qt.basis(2,0)] * N)
    ghz_down = qt.tensor([qt.basis(2,1)] * N)
    psi0 = (ghz_up + ghz_down).unit()
    phi_state = ghz_down  # Renamed to avoid conflict with numerical phi

    neg_grid = np.zeros((len(beta_list), len(strain_list)))
    conc_grid = np.zeros((len(beta_list), len(strain_list)))

    for i, beta in enumerate(beta_list):
        for j, strain_base in enumerate(strain_list):
            strain_amp = strain_base * beta

            def H_t(t, args):
                mod = 1 + strain_amp * np.sin(omega_saw * t)
                return epsilon * (Sz_total + g0 * mod * Sz_total**2)

            c_ops = []
            for k in range(N):
                ops = [qt.qeye(2)] * N
                ops[k] = qt.sigmam()
                c_ops.append(np.sqrt(gamma1) * qt.tensor(ops))
                if gamma_phi > 0:
                    ops[k] = qt.sigmaz() / np.sqrt(2)
                    c_ops.append(np.sqrt(gamma_phi) * qt.tensor(ops))

            times = np.linspace(0, t_weak, n_steps)
            opts = {'nsteps': 15000, 'rtol': 1e-9, 'atol': 1e-11}
            result = qt.mesolve(H_t, psi0, times, c_ops=c_ops, options=opts)
            rho_final = result.states[-1] * result.states[-1].dag()
            rho_final = rho_final / rho_final.tr()

            neg_grid[i,j] = compute_negentropy(rho_final)
            conc_grid[i,j] = average_concurrence(rho_final, N, n_bipart)

            print(f"β={beta:.4f}, strain={strain_base:.3f} | NegEnt={neg_grid[i,j]:.4f} | C_avg={conc_grid[i,j]:.4f}")

    # Heatmap negentropy
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(neg_grid, cmap='viridis', aspect='auto',
                   extent=[strain_list.min(), strain_list.max(), beta_list.max(), beta_list.min()])
    ax.set_xlabel('strain_amp base')
    ax.set_ylabel(r'$\beta$')
    ax.set_title('Heatmap Negentropy von Neumann')
    fig.colorbar(im, ax=ax, label='Negentropy')
    ax.axvline(x=0.09, color='white', ls='--', lw=1.5)
    ax.axhline(y=beta_aureo, color='gold', ls='--', lw=2, label=r'$\phi^{-2} \approx 0.382$') # Fixed here
    ax.legend()
    plt.tight_layout()
    plt.savefig('heatmap_negentropy_beta_strain.jpg', dpi=400, bbox_inches='tight')
    plt.close()

    # Heatmap concurrence media
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    im2 = ax2.imshow(conc_grid, cmap='plasma', aspect='auto',
                     extent=[strain_list.min(), strain_list.max(), beta_list.max(), beta_list.min()])
    ax2.set_xlabel('strain_amp base')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_title('Heatmap Concurrence media multi-bipartizione')
    fig2.colorbar(im2, ax=ax2, label='C_avg')
    ax2.axvline(x=0.09, color='white', ls='--', lw=1.5)
    ax2.axhline(y=beta_aureo, color='gold', ls='--', lw=2) # Fixed here
    plt.tight_layout()
    plt.savefig('heatmap_concurrence_beta_strain.jpg', dpi=400, bbox_inches='tight')
    plt.close()

    print("Figure salvate:")
    print("  heatmap_negentropy_beta_strain.jpg")
    print("  heatmap_concurrence_beta_strain.jpg")

    return beta_list, strain_list, neg_grid, conc_grid


if __name__ == '__main__':
    beta_list = np.linspace(0.30, 0.45, 16)
    strain_list = np.linspace(0.05, 0.13, 9)
    grid_search_2d_beta_strain(beta_list=beta_list, strain_list=strain_list, N=6)


