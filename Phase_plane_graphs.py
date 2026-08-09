#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:17:36 2022

@author: marc
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from common_functions import *
import common_functions
from scipy.integrate import solve_ivp
import os.path

filename = "poincare_map_chaotic_100000.npz"

globals().update(chaotic)
initialize(chaotic)

if os.path.isfile(filename):
    data = np.load(filename)
    t = data['t']
    y = data['y']
else: 
    transient_integration()
    sol = common_functions.sol
    #intgration time
    t_final_driven = 1e5
    # Poincaré section
    T = 2*np.pi*eps
    t_eval = np.arange(t_final_driven/T)*T

    y0 = [F0(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
          F1(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
          sol.y[2,-1], sol.y[3,-1], sol.y[4,-1]]

    sol = solve_ivp(F, [t_eval[0], t_eval[-1]], y0,
                    rtol=1e-9, atol=1e-9, t_eval=t_eval)

    t = sol.t
    y = sol.y
    np.savez(filename, t=t, y=y)

wplot = w(y[0], y[1], y[2], y[3], y[4], order).T

plot_parameters(r=1, n=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  
ax1.plot(t, E(y[0], y[1], y[2], y[3], y[4], order))
ax1.set_xlabel(r'$t$')
ax1.set_ylabel("Fast Energy")

ax2.plot(wplot[0], wplot[1])
ax2.set_xlabel(r'$q_{\text{fast}}$')
ax2.set_ylabel(r'$p_{\text{fast}}$')

plt.savefig("chaotic_phase_plane.pdf", bbox_inches='tight') #for chaotic

filename = "poincare_map_periodic_50.npz"

globals().update(periodic)
initialize(periodic)

if os.path.isfile(filename):
    data = np.load(filename)
    t = data['t']
    y = data['y']
else:
    transient_integration()
    sol = common_functions.sol
    #intgration time
    t_final_driven = 50
    # Poincaré section
    T = 2*np.pi*eps
    t_eval = np.arange(t_final_driven/T)*T

    y0 = [F0(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
          F1(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
          sol.y[2,-1], sol.y[3,-1], sol.y[4,-1]]

    sol = solve_ivp(F, [t_eval[0], t_eval[-1]], y0,
                    rtol=1e-9, atol=1e-9, t_eval=t_eval)

    t = sol.t
    y = sol.y
    np.savez(filename, t=t, y=y)

wplot = w(y[0], y[1], y[2], y[3], y[4], order).T

plot_parameters(r=1, n=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  
ax1.plot(t, E(y[0], y[1], y[2], y[3], y[4], order))
ax1.set_xlabel(r'$t$')
ax1.set_ylabel("Fast Energy")

ax2.plot(wplot[0], wplot[1])
ax2.set_xlabel(r'$q_{\text{fast}}$')
ax2.set_ylabel(r'$p_{\text{fast}}$')

plt.savefig("periodic_phase_plane.pdf", bbox_inches='tight') #for periodic

plt.show()
