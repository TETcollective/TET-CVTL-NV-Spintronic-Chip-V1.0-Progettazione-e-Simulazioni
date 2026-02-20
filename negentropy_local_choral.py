# code/negentropy_local_choral.py
# Negentropia locale in sistema bipartito (NV + ambiente biologico)
# Genera negentropy_local_choral.jpg

import qutip as qt
from qutip import entropy_vn
import matplotlib.pyplot as plt

dim_A = 3
g_A = qt.basis(dim_A, 0)
p_A = qt.basis(dim_A, 1)

dim_B = 2
up_B = qt.basis(dim_B, 0)
dn_B = qt.basis(dim_B, 1)

rho0 = qt.tensor(qt.ket2dm(g_A + 0.1 * p_A).unit(), qt.ket2dm(up_B + 0.2 * dn_B).unit())

Sz_A = qt.tensor(qt.jmat(1, 'z') * 2, qt.qeye(dim_B))
Sz_B = qt.tensor(qt.qeye(dim_A), qt.sigmaz())
H_int = 0.5 * Sz_A * Sz_B

t = 1.5
U = (-1j * H_int * t).expm()
rho_evolved = U * rho0 * U.dag()

S_A_before = entropy_vn(rho0.ptrace(0))
S_A_after  = entropy_vn(rho_evolved.ptrace(0))

negentropy_gain = S_A_before - S_A_after

print(f"Entropia locale A iniziale: {S_A_before:.4f}")
print(f"Entropia locale A dopo choral: {S_A_after:.4f}")
print(f"Negentropia generata: {negentropy_gain:.4f}")

plt.bar(['Iniziale', 'Dopo choral'], [S_A_before, S_A_after], color=['gray', 'darkred'])
plt.ylabel('Entropia von Neumann locale di A')
plt.title('Generazione di negentropia locale via interazione choral')
plt.tight_layout()
plt.savefig('negentropy_local_choral.jpg', dpi=300)
plt.show()