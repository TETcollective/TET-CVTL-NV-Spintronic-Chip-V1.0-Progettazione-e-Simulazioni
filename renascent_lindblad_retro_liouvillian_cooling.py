#code/renascent_lindblad_retro_liouvillian_cooling.py
# Versione con Liouvillian + bias cooling verso stato puro (riduzione visibile)

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

print("QuTiP version:", qt.__version__)

GHz_to_rad_per_us = 2 * np.pi

N = 3
sz = qt.jmat(1, 'z')
sx = qt.jmat(1, 'x')

D0 = 2.87
B = 0.1025
gfactor = 2.8

H0 = GHz_to_rad_per_us * (D0 * sz**2 + gfactor * B * sz)

gamma_relax = 0.01
gamma_deph  = 0.05   # ridotto per lasciare spazio al cooling retro
c_ops_std = [
    np.sqrt(gamma_relax) * sz,
    np.sqrt(gamma_deph / 2) * sx
]

# Stato futuro target: puro |m_s=0⟩
psi_fut = qt.basis(N, 1)
P_fut   = psi_fut * psi_fut.dag()

gamma_retro = 0.60   # aumentato + bias per cooling netto

# Stato iniziale: thermal alta T (entropia ~1.05 bits)
rho0 = qt.thermal_dm(N, 3.0)

times = np.linspace(0, 5, 1000)  # più punti per precisione

# Liouvillian standard
L_std = qt.liouvillian(H0, c_ops_std)

# Liouvillian retro con bias cooling (enfatizziamo proiezione positiva)
# Approssimazione: + gamma_retro * (P ρ P - 0.2 * {P, ρ}) → più "up" che "down"
L_retro = gamma_retro * qt.sprepost(P_fut, P_fut) - 0.2 * gamma_retro * (qt.spre(P_fut) + qt.spost(P_fut))

L_total = L_std + L_retro

# Solver standard
result_std = qt.mesolve(H0, rho0, times, c_ops=c_ops_std, e_ops=[sz], options={'store_states': True})

# Solver retro
result_retro = qt.mesolve(L_total, rho0, times, c_ops=[], e_ops=[sz], options={'store_states': True})

# Entropia
S_std   = [qt.entropy_vn(state) for state in result_std.states]
S_retro = [qt.entropy_vn(state) for state in result_retro.states]

plt.figure(figsize=(10, 6))
plt.plot(times, S_std, label='Lindblad standard', linewidth=2.5)
plt.plot(times, S_retro, label=f'Con retro cooling (γ_retro = {gamma_retro:.2f} /μs)', linewidth=2.5)
plt.xlabel('Tempo (μs)')
plt.ylabel('Entropia von Neumann S(ρ) [bits]')
plt.title('Riduzione locale di entropia via termine retrocausale cooling\n(β ≈ 0.382)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('renascent_entropy_cooling.png', dpi=300)
plt.show()

print(f"Entropia iniziale:         {qt.entropy_vn(rho0):.4f} bits")
print(f"Entropia finale standard:  {S_std[-1]:.4f} bits")
print(f"Entropia finale con retro: {S_retro[-1]:.4f} bits")
print(f"Delta S (riduzione):       {S_std[-1] - S_retro[-1]:+.4f} bits")