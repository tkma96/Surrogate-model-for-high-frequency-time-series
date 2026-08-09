#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:20:21 2023

@author: marc
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from common_functions import *
import common_functions
from scipy.integrate import solve_ivp

initialize(chaotic)
transient_integration()
globals().update(chaotic)
sol = common_functions.sol

t_final_driven = 7
t_eval = np.linspace(0, t_final_driven, 1000)
    
y0 = [F0(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
      F1(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
      sol.y[2,-1], sol.y[3,-1], sol.y[4,-1]]

sol = solve_ivp(F, [t_eval[0], t_eval[-1]], y0,
                rtol=1e-11, atol=1e-11, t_eval=t_eval)

# The AAA approximation
r_z = aaa(sol.t, sol.y[4], tol=1e-14, return_errors=False)
r_y = aaa(sol.t, sol.y[3], tol=1e-14, return_errors=False)
r_x = aaa(sol.t, sol.y[2], tol=1e-14, return_errors=False)

poles_x, res_x = r_x.polres(use_mp=False)
poles_y, res_y = r_y.polres(use_mp=False)
poles_z, res_z = r_z.polres(use_mp=False)

plot_parameters(r=1,n=0.76)

fig, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout = True)

ax1.scatter(poles_x.real, poles_x.imag, c='#2ca02c',
            marker='x', s=2*np.abs(res_x), label='$x$-pole')
ax1.scatter(poles_y.real, poles_y.imag, c='#ff7f0e',
            marker='+', s=2*np.abs(res_y), label='$y$-pole')
ax1.scatter(poles_z.real, poles_z.imag, 
            marker='o', s=2*np.abs(res_z), label='$z$-pole',
            facecolors='none', edgecolors='#1f77b4',
            linewidths=0.3)

ax1.set_ylabel(r'$\operatorname{Im}(t)$')
ax1.set_xlim(t_eval[0], t_eval[-1])
ax1.set_ylim(-1, 1)
ax1.grid(color='k', linestyle='--', linewidth=0.2)
lgnd = ax1.legend(loc='lower right', ncol=3)
# Fix size of markers in legend
# See https://stackoverflow.com/a/43578952
for handle in lgnd.legendHandles:
    handle.set_sizes([8.0])

ax2.plot(sol.t,
         E(sol.y[0], sol.y[1], sol.y[2],
           sol.y[3], sol.y[4], order))
ax2.set_ylabel(r'Fast Energy')
ax2.set_xlim(t_eval[0], t_eval[-1])
ax2.grid(color='k', linestyle='--', axis='x' ,linewidth=0.2)

ax3.plot(sol.t, sol.y[4])
ax3.set_ylabel('$z(t)$')
ax3.set_xlim(t_eval[0], t_eval[-1])
ax3.set_xlabel(r'$\operatorname{Re}(t)$')

plt.savefig("fast_energy_level_pole_location.pdf", bbox_inches='tight')
plt.show()
