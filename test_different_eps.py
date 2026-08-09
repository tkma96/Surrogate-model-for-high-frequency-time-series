#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from parameters import *
from common_functions import *
import os.path

filename = "eps05_mmax9.npz"
chaotic['eps'] = 0.05
chaotic['order'] = 0
chaotic['mmax'] = 9

if os.path.isfile(filename):
    data = np.load(filename)
    E_num = data['E_num']
    B_sur = data['B_sur']
else: 
    initialize(chaotic)
    E_num, B_sur = simulation(N=10000)
    np.savez(filename, E_num=E_num, B_sur=B_sur)

surrogate = np.abs(np.cumsum(B_sur))

plt.figure()
plt.acorr(B_sur, usevlines=True, normed=True, maxlags=30)
plt.title("Autocorrelation of complex connection amplitude")

plt.figure()
plt.acorr(np.sqrt(E_num) - np.abs(surrogate),
          usevlines=True, normed=True, maxlags=30)
plt.title("Autocorrelation of error")

plt.figure()
plt.plot(np.sqrt(E_num) - np.abs(surrogate))
plt.title("Error")

#plot_parameters(r=1, n=1.7)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.plot(np.sqrt(E_num), label="Full Model")
ax1.plot(abs(surrogate), label="Surrogate Model")
ax1.set_xlabel(r'$k$')
ax1.set_ylabel("Fast Amplitude")
ax1.legend()

ax2.plot(MSD(np.sqrt(E_num)), label="Full Model")
ax2.plot(MSD(abs(surrogate)), label="Surrogate Model")
ax2.legend()
ax2.set_ylabel(r'$\operatorname{MSD}(n)$')
ax2.set_xlabel(r'$n$')

#plt.savefig("comparison_chaotic.pdf",  bbox_inches='tight')
plt.show()

