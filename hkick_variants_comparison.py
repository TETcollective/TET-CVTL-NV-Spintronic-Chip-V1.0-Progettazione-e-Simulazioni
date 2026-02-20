"""
Confronto Varianti H_kick per RENASCENT-Q
========================================
Confronto tra ZZ only, XY only e XX+ZZ (gold) per protezione entanglement sotto decoerenza.
Output: Figura hkick_variants_comparison.pdf con tre curve concurrence vs tempo.



import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def hkick_variants_comparison(
    gamma_relax=0.012,
    t_braid=10.0,
    kick_interval=4.0,
    kick_strength=5.0,
    kick_duration=0.5
):
    gamma_deph = gamma_relax * 0.6

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

    opts = {'store_states': True, 'nsteps': 12000, 'rtol': 1e-8, 'atol': 1e-10}

    variants = {
        'ZZ only': kick_strength * qt.tensor(qt.sigmaz(), qt.sigmaz()),
        'XY only': kick_strength * (qt.tensor(qt.sigmax(), qt.sigmay()) +
                                    qt.tensor(qt.sigmay(), qt.sigmax())),
        'XX+ZZ (gold)': kick_strength * (qt.tensor(qt.sigmax(), qt.sigmax()) +
                                         0.4 * qt.tensor(qt.sigmaz(), qt.sigmaz()))
    }

    results = {}
    plt.figure(figsize=(13, 7.5))

    for name, H_kick in variants.items():
        print(f"Esecuzione: {name}")
        current = psi0
        all_states = [current]
        all_times = [0.0]

        for ks in np.arange(kick_interval, t_braid + 1e-6, kick_interval):
            # Evoluzione libera
            t_end = min(ks, t_braid)
            if t_end - all_times[-1] > 0.02:
                t_norm = np.linspace(all_times[-1], t_end, 200)
                res = qt.mesolve(H_main, current, t_norm, c_ops=c_ops, options=opts)
                current = res.states[-1]
                all_states.extend(res.states[1:])
                all_times.extend(t_norm[1:])

            # Kick
            if ks + kick_duration <= t_braid:
                t_kick = np.linspace(ks, ks + kick_duration, 35)
                res_k = qt.mesolve(H_kick, current, t_kick, c_ops=c_ops, options=opts)
                current = res_k.states[-1]
                all_states.extend(res_k.states[1:])
                all_times.extend(t_kick[1:])

        # Ultimo tratto
        if all_times[-1] < t_braid - 0.02:
            t_final = np.linspace(all_times[-1], t_braid, 200)
            res_final = qt.mesolve(H_main, current, t_final, c_ops=c_ops, options=opts)
            all_states.extend(res_final.states[1:])
            all_times.extend(t_final[1:])

        conc_t = [qt.concurrence(s * s.dag() if s.isket else s) for s in all_states]
        final_c = conc_t[-1]
        results[name] = final_c

        plt.plot(all_times, conc_t, label=f'{name}  (C={final_c:.4f})', lw=2.2, alpha=0.92)

    plt.xlabel('Tempo (unità arbitrarie)')
    plt.ylabel('Concurrence')
    plt.title(f'Confronto varianti H_kick — γ_relax = {gamma_relax:.3f}')
    plt.grid(True, alpha=0.3, ls='--')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('hkick_variants_comparison.pdf', dpi=380, bbox_inches='tight')
    plt.show()

    print("\nRisultati finali:")
    for name, val in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:18} → {val:.4f}")

    return results


if __name__ == '__main__':
    hkick_variants_comparison()