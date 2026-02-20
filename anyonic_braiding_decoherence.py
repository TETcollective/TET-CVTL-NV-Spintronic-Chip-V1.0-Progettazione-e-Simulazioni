# code/anyonic_braiding_decoherence.py



import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

gamma_relax = 0.015
gamma_deph  = 0.008
t_braid     = 10.0
n_steps     = 200
theta_phase = np.pi / 3

bell_plus = (qt.basis(4,0) + qt.basis(4,3)).unit()
psi0 = bell_plus

H_phase = theta_phase * qt.basis(4,3).proj()
swap_like = qt.Qobj(np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
]), dims=[[4], [4]])
H = H_phase + 0.8 * swap_like

sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))
sm2 = qt.tensor(qt.qeye(2), qt.sigmam())
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

c_ops = [
    np.sqrt(gamma_relax) * sm1,
    np.sqrt(gamma_relax) * sm2,
    np.sqrt(gamma_deph / 2) * sz1,
    np.sqrt(gamma_deph / 2) * sz2
]

times = np.linspace(0, t_braid, n_steps)

result = qt.mesolve(H, psi0, times, c_ops)

concurrence_t = []
entropy_t = []

for state in result.states:
    rho_bipart = qt.Qobj(state.full(), dims=[[2,2], [2,2]])
    concurrence_t.append(qt.concurrence(rho_bipart))
    entropy_t.append(qt.entropy_vn(state))

concurrence_final = concurrence_t[-1]
entropy_final = entropy_t[-1]

print(f"Concurrence finale: {concurrence_final:.4f}")
print(f"Entropy finale: {entropy_final:.4f}")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(times, concurrence_t, 'o-', color='purple', label='Concurrence embodied')
ax1.set_xlabel('Tempo')
ax1.set_ylabel('Concurrence embodied')
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(times, entropy_t, 's--', color='orange', label='Entropy von Neumann')
ax2.set_ylabel('Entropy')

plt.title('Anyonic braiding con decoerenza Lindblad')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('anyonic-braiding-decoherence-plot.jpg', dpi=300)
plt.show()