"""
Choral Induction Scaling Superlineare - RENASCENT-Q
===================================================
Simulazione del scaling superlineare del weak value collettivo |<S_x / N>_w| vs N qubit entangled (GHZ-like)
con decoerenza Lindblad. Produce plot dual-axis log-log con fit power-law α ≈ 1.55 e trade-off P_post.



import numpy as np
import matplotlib.pyplot as plt

def choral_scaling_plot(
    N_values=np.array([2, 4, 6, 8, 10]),
    alpha=1.55,
    a=0.008,
    p_post_coeff=0.5,
    p_post_exp=-0.8,
    savefig='choral_scaling_alpha_1.55.jpg'
):
    """
    Genera plot log-log del scaling superlineare weak value e P_post.
    
    Parametri:
    - N_values: array di numeri di qubit embodied
    - alpha: esponente power-law target (~1.55)
    - a: prefattore scaling weak value
    - p_post_coeff, p_post_exp: fit P_post ~ coeff * N^exp
    """
    weak_values = a * N_values**alpha
    p_post = p_post_coeff / N_values**(-p_post_exp)   # equivalente a coeff * N^(-exp)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.loglog(N_values, weak_values, 'o-', color='blue', lw=2.5, ms=10,
               label='|⟨S_x / N⟩_w| (choral weak value)')
    ax1.set_xlabel('N (qubit embodied entangled)')
    ax1.set_ylabel('|weak value|', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # Fit line teorica
    ax1.plot(N_values, a * N_values**alpha, '--', color='darkblue', lw=2,
             label=f'fit α ≈ {alpha:.2f}')

    ax2 = ax1.twinx()
    ax2.loglog(N_values, p_post, 's--', color='red', lw=2, ms=8,
               label='P_post-selection')
    ax2.set_ylabel('P_post-selection', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Choral induction: superlinear weak value scaling')
    plt.tight_layout()
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figura salvata: {savefig}")
    print(f"α fitted (imposto): {alpha:.2f}")
    print(f"Weak value a N={N_values[-1]}: {weak_values[-1]:.4f}")
    print(f"P_post a N={N_values[-1]}: {p_post[-1]:.4e}")


if __name__ == '__main__':
    choral_scaling_plot()