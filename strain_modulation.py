# code/strain_modulation.py
epsilon_par = np.linspace(-0.001, 0.001, 100)  # strain relativo
dD_de = -14.6e9  # Hz / %
D_strained = D + dD_de * epsilon_par

# Simulazione ODMR shift o coherence