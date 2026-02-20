# code/weak_value_retrocasual_nv.py



import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

g = qt.basis(3, 0)  # |0>
p = qt.basis(3, 1)  # |+1>
m = qt.basis(3, 2)  # |-1>

Sz = qt.jmat(1, 'z') * 2

psi_pre = (g + 0.5 * p - 0.5 * m).unit()

leak = 0.0001
phi_post = (p + leak * g).unit()

A = Sz

wv_classic_raw = psi_pre.dag() * A * psi_pre
wv_classic = wv_classic_raw[0,0] if hasattr(wv_classic_raw, '__getitem__') else complex(wv_classic_raw)

num_raw = phi_post.dag() * A * psi_pre
den_raw = phi_post.dag() * psi_pre

num = num_raw[0,0] if hasattr(num_raw, '__getitem__') else complex(num_raw)
den = den_raw[0,0] if hasattr(den_raw, '__getitem__') else complex(den_raw)

wv_retro = num / (den + 1e-12)

print(f"Weak value classico: {wv_classic:.4f}")
print(f"Weak value retrocausale: {wv_retro:.4f}")
print(f"Amplificazione: {abs(wv_retro) / abs(wv_classic):.2f}x")

plt.figure(figsize=(6, 5))
plt.bar(['Classico', 'Retrocausale'], [abs(wv_classic), abs(wv_retro)], color=['gray', 'darkred'], width=0.6)
plt.ylabel('|Weak Value|', fontsize=12)
plt.title('Amplificazione retrocausale del weak value in NV-like')
plt.ylim(0, max(2, abs(wv_retro)*1.2))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('weak_value_retrocausal_nv.jpg', dpi=300)
plt.show()