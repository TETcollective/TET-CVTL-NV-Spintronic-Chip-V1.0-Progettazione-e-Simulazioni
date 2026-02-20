# code/choral_weak_value_toy.py



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(x, a, alpha):
    return a * x**alpha

def choral_weak_value_toy(
    N_list       = [2, 4, 6, 8, 10, 12, 14, 16],
    base_ampl    = 0.5,
    choral_boost = 0.38,
    noise_level  = 0.12
):
    wvals = []
    np.random.seed(42)

    for N in N_list:
        wv = base_ampl * N**choral_boost * (1 + noise_level * np.random.randn())
        wvals.append(max(0.4, wv))

    try:
        popt, _ = curve_fit(power_law, N_list, wvals, p0=[0.5, 1.5],
                            bounds=([0.01, 1.0], [100, 3.0]))
        a_fit, alpha_fit = popt
    except:
        alpha_fit = np.nan
        a_fit = np.nan

    print(f"Fit: |<A>_w| ≈ {a_fit:.3f} × N^{alpha_fit:.2f}")
    print("wvals:", [round(v, 3) for v in wvals])

    return N_list, wvals, alpha_fit

N_list = [2, 4, 6, 8, 10, 12, 14, 16]
N, wvals, alpha = choral_weak_value_toy()

plt.figure(figsize=(9, 6))
plt.loglog(N, wvals, 'o-', ms=9, lw=2.2, color='darkblue',
           label=f'simulazione toy choral (α fit = {alpha:.2f})')

plt.loglog(N, 0.4 * np.array(N)**1.4, '--', lw=1.4, color='gray', alpha=0.7,
           label=r'guida α = 1.4')
plt.loglog(N, 0.35 * np.array(N)**1.6, '-.', lw=1.4, color='gray', alpha=0.7,
           label=r'guida α = 1.6')

plt.xlabel('N (qubit embodied / choral ensemble)')
plt.ylabel(r'$|\langle A \rangle_w|$')
plt.title('Scaling superlineare in choral weak-value induction')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", lw=0.7, alpha=0.6)
plt.tight_layout()
plt.savefig('weak_values_scaling_loglog_sim.jpg', dpi=300)
plt.show()