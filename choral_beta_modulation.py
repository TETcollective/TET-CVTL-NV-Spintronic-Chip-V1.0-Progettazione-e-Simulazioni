"""
Modulazione Aurea β per Choral Induction - RENASCENT-Q
======================================================
Script per visualizzare modulazione strain_amp e g tramite β = φ⁻².
Include plot esemplificativi di strain(t) e g(t) con oscillazione aurea.



import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2
beta = 1 / phi**2  # ≈ 0.381966

def plot_beta_modulation(
    t_max=10.0,
    n_points=1000,
    delta=0.05,
    kappa=0.1,
    f_mod=1.0,
    strain0=0.09,
    g0=0.17,
    savefig_strain='strain_beta_modulation.png',
    savefig_g='g_beta_modulation.png'
):
    """
    Plot strain_amp(t) e g(t) modulati attorno a β.
    """
    t = np.linspace(0, t_max, n_points)

    strain_t = strain0 * (beta + delta * np.sin(2 * np.pi * f_mod * t))
    g_t = g0 * (1 + kappa * (beta - beta))   # qui costante, ma può oscillare

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, strain_t, 'b-', lw=2.5)
    ax1.axhline(y=strain0 * beta, color='gold', ls='--', lw=2, label=r'$\beta \cdot$ strain₀')
    ax1.set_xlabel('Tempo (unità arbitrarie)')
    ax1.set_ylabel('strain_amp(t)')
    ax1.set_title('Modulazione strain SAW attorno a β aureo')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(savefig_strain, dpi=300, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t, g_t * np.ones_like(t), 'g-', lw=2.5)  # esempio costante
    ax2.axhline(y=g0, color='gold', ls='--', lw=2, label=r'g₀')
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('g(t)')
    ax2.set_title('g(t) con bias aureo β')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(savefig_g, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure salvate: {savefig_strain}, {savefig_g}")
    print(f"β aureo utilizzato: {beta:.6f}")


if __name__ == '__main__':
    plot_beta_modulation()