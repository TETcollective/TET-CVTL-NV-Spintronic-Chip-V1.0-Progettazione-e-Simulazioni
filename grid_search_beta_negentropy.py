"""
Grid Search su β per Picco Negentropia - RENASCENT-Q
===================================================
Grid search 1D su β ∈ [0.36, 0.40] per N=4 qubit GHZ-like.
Calcola negentropy von Neumann e negativity multipartita.
Mostra picco vicino a φ⁻² ≈ 0.382.



import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2
beta_aureo = 1 / phi**2

def compute_negentropy(rho):
    """Negentropy = -S_vN (entropia von Neumann negativa)"""
    if rho.type == 'ket':
        rho = rho * rho.dag()
    evals = rho.eigenenergies()
    evals = evals[evals > 1e-14]
    if len(evals) == 0:
        return 0.0
    S = -np.sum(evals * np.log(evals + 1e-20))
    return -S


def ghz_negativity(rho, N):
    """Negativity approssimata per stato GHZ multipartita (PT manuale su metà sistema)"""
    if N < 2:
        return 0.0
    cut = N // 2
    subsA = list(range(cut))
    
    # Partial trace sul complemento per ottenere rho_AB
    subsB = list(range(cut, N))
    rho_ab = rho.ptrace(subsA + subsB)
    
    dimA = 2**cut
    dimB = 2**cut
    
    rho_mat = rho_ab.full()
    shape = (dimA, dimB, dimA, dimB)
    rho_tensor = rho_mat.reshape(shape)
    
    # Partial transpose su A: swap assi 0 e 2
    rho_pt_tensor = np.transpose(rho_tensor, (2, 1, 0, 3))
    rho_pt_mat = rho_pt_tensor.reshape((dimA*dimB, dimA*dimB))
    rho_pt = qt.Qobj(rho_pt_mat, dims=rho_ab.dims)
    
    evals = rho_pt.eigenenergies()
    negativity = (np.sum(np.abs(evals)) - 1) / 2
    return max(0, negativity)


def grid_search_beta(
    beta_min=0.36,
    beta_max=0.40,
    n_beta=41,
    N=4,
    t_weak=0.18,
    gamma1=1.0/500.0,
    gamma_phi=1.0/200.0 - 0.5/500.0,
    savefig='grid_search_peak_final.jpg'
):
    beta_list = np.linspace(beta_min, beta_max, n_beta)

    # Operatori collettivi
    Sz_total = sum(qt.tensor([qt.qeye(2)]*i + [qt.sigmaz()] + [qt.qeye(2)]*(N-1-i)) for i in range(N))

    # Stato GHZ iniziale
    ghz_up   = qt.tensor([qt.basis(2,0)] * N)
    ghz_down = qt.tensor([qt.basis(2,1)] * N)
    psi0 = (ghz_up + ghz_down).unit()

    negentropies = []
    negativities = []

    for beta in beta_list:
        strain_amp = 0.04 * beta
        g = 0.10

        def H_t(t, args):
            mod = 1 + strain_amp * np.sin(2 * np.pi * 4.0 * t)
            return 0.002 * (Sz_total + g * mod * Sz_total**2)

        c_ops = []
        for i in range(N):
            ops = [qt.qeye(2)] * N
            ops[i] = qt.sigmam()
            c_ops.append(np.sqrt(gamma1) * qt.tensor(ops))
            if gamma_phi > 0:
                ops[i] = qt.sigmaz() / np.sqrt(2)
                c_ops.append(np.sqrt(gamma_phi) * qt.tensor(ops))

        times = np.linspace(0, t_weak, 1200)
        opts = {'nsteps': 25000, 'rtol': 1e-9, 'atol': 1e-11}
        result = qt.mesolve(H_t, psi0, times, c_ops=c_ops, options=opts)
        rho_final = result.states[-1]

        neg_S = compute_negentropy(rho_final)
        negentropies.append(neg_S)

        neg = ghz_negativity(rho_final, N)
        negativities.append(neg)

        print(f"β={beta:.5f} | NegEnt={neg_S:.4f} | Neg={neg:.4f}")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(beta_list, negentropies, 'o-', color='green', lw=2.8, ms=9,
             label='Negentropy (bits)')
    ax1.set_xlabel('β')
    ax1.set_ylabel('Negentropy von Neumann', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.axvline(x=beta_aureo, color='gold', ls='--', lw=3.5, label=r'$\phi^{-2} \approx 0.382$')
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(beta_list, negativities, 's--', color='blue', lw=2.2,
             label='Negativity multipartita')
    ax2.set_ylabel('Negativity', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.suptitle(f'Grid search RENASCENT-Q: picco negentropy vs β (N={N})')
    fig.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(savefig, dpi=500, bbox_inches='tight')
    plt.show()

    idx_max = np.argmax(negentropies)
    beta_opt = beta_list[idx_max]
    print(f"\nBeta ottimale per max negentropy: {beta_opt:.6f}")
    print(f"Negentropy max: {negentropies[idx_max]:.4f}")
    print(f"Negativity al picco: {negativities[idx_max]:.4f}")
    print(f"Distanza da φ⁻²: {abs(beta_opt - beta_aureo):.6f}")


if __name__ == '__main__':
    grid_search_beta()