# code/nv_ground_hamiltonian.py
import qutip as qt
import numpy as np

# Costanti
D = 2.870e9  # Hz
g_e = 2.0028
mu_B = 9.274e-24 / (6.626e-34) * 1e-9  # approx 28 GHz/T in Hz/T
A_par = 2.14e6   # Hz for 14N
A_perp = -2.70e6
P = -4.95e6      # quadrupole 14N

# Spin operators S=1, I=1 for 14N
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')
Sz = qt.jmat(1, 'z')
Ix = qt.jmat(1, 'x')
Iy = qt.jmat(1, 'y')
Iz = qt.jmat(1, 'z')

# Hamiltonian terms
H_ZFS = D * (Sz**2 - 2/3 * qt.qeye(3))
H_Zeeman = lambda Bz: g_e * mu_B * Bz * Sz   # Bz in T
H_hyperfine = A_par * Sz * Iz + A_perp * (Sx * Ix + Sy * Iy)
H_quad = P * (Iz**2 - 2/3 * qt.qeye(3))   # approx for quadrupole

# Full H for Bz = 0
H0 = H_ZFS + H_hyperfine + H_quad

# Esempio: eigenvalues per check splitting
eigs = H0.eigenenergies() / 1e9  # in GHz
print("Eigenenergies (GHz):", eigs)