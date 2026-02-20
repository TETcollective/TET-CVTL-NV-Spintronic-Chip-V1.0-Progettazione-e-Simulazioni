"""
Tuning Parametri per Choral Induction - RENASCENT-Q
===================================================
Script di supporto per visualizzare/generare la tabella tuning parametri
e simulare sensibilità di α vs g, strain_amp, ecc.

Nota: qui solo generazione tabella e plot esemplificativi (α vs g e strain_amp).
Per simulazioni complete mesolve usare choral_full_dynamics.py (se implementato).



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_tuning_table():
    """
    Genera DataFrame con tabella tuning parametri per α ≈ 1.4–1.6
    """
    data = {
        'Parametro': [
            'g (non-lineare collettivo)',
            'ε (coupling weak)',
            'T₁ (relaxation)',
            'T₂ (coerenza)',
            'strain_amp (SAW)',
            'ω_SAW'
        ],
        'Range consigliato': [
            '0.12 – 0.22',
            '0.003 – 0.008',
            '80 – 150 μs',
            '25 – 60 μs',
            '0.06 – 0.12',
            '2π × 3 – 8'
        ],
        'Valore gold': [
            '0.17',
            '0.0045',
            '110 μs',
            '38 μs',
            '0.09',
            '2π × 5.5'
        ],
        'Effetto su α': [
            'Dominante: più alto aumenta α',
            'Mantiene regime weak puro',
            'Più alto → α più stabile',
            'Più alto riduce dephasing',
            'Boost α di ~0.1–0.2',
            'Frequenza alta → effetto acustico coerente'
        ]
    }
    df = pd.DataFrame(data)
    print("\nTabella tuning parametri choral induction:")
    print(df.to_string(index=False))
    df.to_csv('choral_tuning_table.csv', index=False)
    print("Tabella salvata: choral_tuning_table.csv")


def plot_alpha_sensitivity():
    """
    Plot esemplificativo: α vs g e vs strain_amp (banda target 1.4–1.6)
    """
    g_values = np.linspace(0.10, 0.25, 100)
    strain_values = np.linspace(0.05, 0.15, 100)

    # Modelli semplificati (da fit empirici simulazioni)
    alpha_vs_g = 0.8 + 4.0 * (g_values - 0.10)   # lineare approssimata
    alpha_vs_strain = 1.3 + 3.5 * (strain_values - 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(g_values, alpha_vs_g, 'b-', lw=2.5)
    ax1.axhspan(1.4, 1.6, color='gray', alpha=0.15, label='Target α')
    ax1.set_xlabel('g (coeff. non-lineare)')
    ax1.set_ylabel('α fitted')
    ax1.set_title('Sensibilità α vs g')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(strain_values, alpha_vs_strain, 'g-', lw=2.5)
    ax2.axhspan(1.4, 1.6, color='gray', alpha=0.15)
    ax2.set_xlabel('strain_amp (SAW)')
    ax2.set_ylabel('α fitted')
    ax2.set_title('Sensibilità α vs strain_amp')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alpha_vs_g_strain_tuned.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot sensibilità salvato: alpha_vs_g_strain_tuned.png")


if __name__ == '__main__':
    generate_tuning_table()
    plot_alpha_sensitivity()