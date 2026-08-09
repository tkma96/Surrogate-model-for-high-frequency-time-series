#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from parameters import *
from common_functions import *


import os.path

filename = "chaotic_rebalanced_20000.npz"

if os.path.isfile(filename):
    data = np.load(filename)
    E_num = data['E_num']
    B_sur = data['B_sur']
else: 
    initialize(chaotic)
    transient_integration()
    E_num, B_sur = simulation(N=20000, k_balance=np.inf)
    np.savez(filename, E_num=E_num, B_sur=B_sur)

rel_difference = (np.sqrt(E_num) - np.abs(B_sur))/max(np.sqrt(E_num))

plot_parameters()

plt.figure()
plt.plot(np.sqrt(E_num), label="Coupled Model")
plt.plot(np.abs(B_sur), label="Connection Formula")
plt.xlabel(r'$m$')
plt.ylabel("Fast Amplitude")
plt.savefig("connection_formula.pdf", bbox_inches='tight')

plt.figure()
plt.hist(rel_difference, 100, color='green', lw=3, log=True,
         weights=np.ones(len(rel_difference)) / len(rel_difference))
plt.xlim(-0.01,0.06)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.xlabel("Relative difference")
plt.ylabel("Frequency")
plt.savefig("Relative_difference_histogram.pdf",  bbox_inches='tight')
plt.show()

print("Mean of relative difference:", np.mean(rel_difference))
print("Standard deviation of relative difference:", np.std(rel_difference))
