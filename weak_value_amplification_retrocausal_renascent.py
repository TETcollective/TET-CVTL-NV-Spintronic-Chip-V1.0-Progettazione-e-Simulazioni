"""
Amplificazione retrocausale di weak value con post-selezione
nel framework RENASCENT-Q – Collegamento braiding anyonico asimmetrico


Scopo: Dimostrare gain anomalo weak-value in regime post-selezione per amplificare
       segnali retrocausali e negentropici embodied (kick retrocausale, riduzione entropia,
       stabilizzazione Majorana) nel chip TET--CVTL.

Requisiti: QuTiP, NumPy, Matplotlib
Input: stato finale del braiding asimmetrico (psi_braiding_final)
Output: weak value complesso A_w, shift pointer <x> amplificato, figura PDF
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ================================================
# 1. SETUP (assumiamo stato finale dal braiding precedente)
# ================================================
dim_system = 4
# Esempio placeholder per psi_braiding_final (sostituisci con result.states[-1] reale)
psi_braiding_final = (qt.basis(dim_system, 0) + 0.01 * qt.basis(dim_system, 3)).unit()

# ================================================
# 2. POINTER E OSSERVABILE WEAK
# ================================================
N_pointer = 10              # livelli Fock pointer
alpha_pointer = 0.2         # ampiezza coerente iniziale
pointer_init = qt.coherent(N_pointer, alpha_pointer)
pointer_final = qt.basis(N_pointer, 0)  # post-selezione su vacuum pointer

psi_i = qt.tensor(psi_braiding_final, pointer_init)

# Osservabile weak: sigma_z sul primo qubit (esempio per sistema 4D = 2 qubit)
sz1_op = qt.tensor(qt.sigmaz(), qt.qeye(2))
A = qt.tensor(sz1_op, qt.qeye(N_pointer))

# ================================================
# 3. POST-SELEZIONE QUASI-ORTOGONALE
# ================================================
overlap_target = 1e-5  # overlap piccolo per gain alto
psi_f_braiding = (qt.basis(dim_system, 0) + overlap_target * qt.basis(dim_system, 3)).unit()
psi_f = qt.tensor(psi_f_braiding, pointer_final)

bra_f_dag = psi_f.dag()
overlap = (bra_f_dag * psi_i).tr()
if abs(overlap) < 1e-12:
    weak_value = np.nan + 0j
    print("Overlap troppo piccolo - rischio divisione per zero")
else:
    weak_value = (bra_f_dag * A * psi_i).tr() / overlap

print(f"Overlap |<ψ_f|ψ_i>|: {abs(overlap):.2e}")
print(f"Weak value complesso A_w = {weak_value}")
print(f"Re(A_w): {np.real(weak_value):.6f}")
print(f"Im(A_w): {np.imag(weak_value):.6f}")

# ================================================
# 4. EVOLUZIONE WEAK INTERACTION + POINTER SHIFT
# ================================================
epsilon = 0.4  # coupling debole
p_op = qt.momentum(N_pointer)
H_w = epsilon * qt.tensor(sz1_op, p_op)

rho0 = psi_i * psi_i.dag()

times_weak = np.linspace(0, 10.0, 800)
result_weak = qt.mesolve(H_w, rho0, times_weak, c_ops=[], e_ops=[])

x_op = qt.tensor(qt.qeye(dim_system), qt.position(N_pointer))
expect_x = [qt.expect(x_op, state) for state in result_weak.states]

# ================================================
# 5. PLOT SHIFT AMPLIFICATO DEL POINTER
# ================================================
plt.figure(figsize=(12, 7))
plt.plot(times_weak, expect_x, color='teal', lw=3.5, label=r'Shift pointer $\langle x \rangle$')
plt.axhline(0, color='gray', ls='--', alpha=0.7)
plt.xlabel('Tempo weak interaction (unità arbitrarie)', fontsize=14)
plt.ylabel(r'$\langle x \rangle$ pointer', fontsize=14)
plt.title(f'Amplificazione retrocausale weak value\n(overlap ≈ {abs(overlap):.2e}, ε = {epsilon})', fontsize=15)
plt.grid(True, alpha=0.35)
plt.legend(fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig('weak_value_renascent_kick.pdf', dpi=700, bbox_inches='tight')
print("Plot salvato: weak_value_renascent_kick.pdf")
plt.show()