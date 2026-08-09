#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sym
from baryrat import aaa
from matplotlib.pyplot import rcParams
from parameters import *

### Define coupled system and computate expressions for the fast
### vector field and for the fast energy

def F(t, vec):
    return np.r_[vec[1]/eps,
                 (vec[4]-vec[0])/eps,
                 sigma*(vec[3]-vec[2]),
                 r*vec[2] - vec[3] - vec[2]*vec[4],
                 vec[2]*vec[3] - b*vec[4]]

def initialize(params):
    global H, sol
    globals().update(params)

    # Analytical derivatives of z(t)

    t,X,Y,Z = sym.symbols('t X Y Z')
    x = sym.Function('x')(t)
    y = sym.Function('y')(t)
    z = sym.Function('z')(t)

    f1 = sigma*(y-x)
    f2 = r*x - y - x*z
    z1 = x*y - b*z
    G = [z]

    for i in range(order):
        p = sym.diff(G[-1],t)
        p = p.xreplace({sym.diff(x,t):f1,
                        sym.diff(y,t):f2,
                        sym.diff(z,t):z1})
        G.append(sym.horner(p))

    H = [sym.lambdify((x,y,z), g) for g in G]

    print("Precomputation is done.")

### Compute an initial transient trajectory so that Lorenz system is
### practically on the attractor

def transient_integration(t_final_transient = 30):
    global sol
    sol = solve_ivp(F, [0, t_final_transient], y0, rtol=1e-9, atol=1e-9)

    print("Transient integration is done.")

def F0(X,Y,Z,n):
    terms = [eps**i * (-1)**(i//2) * H[i](X,Y,Z) for i in range(0,n+1,2)]
    return sum(terms)

def F1(X,Y,Z,n):
    terms = [eps**i * (-1)**(i//2) * H[i](X,Y,Z) for i in range(1,n+1,2)]
    return sum(terms)

def E(q,p,X,Y,Z,n):
    return (q-F0(X,Y,Z,n))**2 + (p-F1(X,Y,Z,n))**2

def w(q,p,X,Y,Z,n):
    return np.c_[q-F0(X,Y,Z,n), p-F1(X,Y,Z,n)]

### Define events for cutting single branch points out of the
### trajectory

def event_up (t,vec):
    return vec[4] - threshold

event_up.terminal = True
event_up.direction = 1

def event_down (t,vec):
    return vec[4] - threshold

event_down.terminal = True
event_down.direction = -1

# We need a final time to pass to the integrator.  However, the
# forward integration will be terminated by the up/down events in
# practice, so the exact value of this variable will not matter
t_final_dummy = 1000000000000

def simulation(N, k_balance=1):
    global sol
    """
    N: number of z-excursions to record
    k_balance: index of event up to which we rebalance the model.
               Default 1 corresponds to the actual surrogate model
               simulation, set to inf to rebalance after every z-excursion

    Returns:
    E_num: Fast energy from direct numerical simulation
    B_sur: Complex jump amplitude from connection formula (surrogate model)
    """
    
    # Go to first up-event for a good start
    sol = solve_ivp(F, [0, t_final_dummy], sol.y[:,-1],
                    events=[event_up], rtol= 1e-9, atol= 1e-9)
     
    E_num = np.zeros(N)
    B_sur = np.zeros(N, dtype=complex)
    difference = np.zeros(N)

    for k in range(N):
        print(k)

        if k < k_balance:
            # Balanced initialization
            print("reinitialize")
            y0 = [F0(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
                  F1(sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], order),
                  sol.y[2,-1], sol.y[3,-1], sol.y[4,-1]]
        else:
            # Start where previous integration left off
            y0 = sol.y[:,-1]

        # At the moment we do not care about the actual time, so let's
        # just restart simulation time at zero.  If we really need to
        # reconstruct the full trajectory, not just the per-event
        # information, we can be more subtle
        sol = solve_ivp(F, [sol.t[-1], t_final_dummy], y0,
                        events=[event_down], rtol= 1e-9, atol= 1e-9) 

        E_num[k] = E(sol.y[0,-1], sol.y[1,-1], sol.y[2,-1],
                     sol.y[3,-1], sol.y[4,-1], order)
    
        # Compute rational interpolant of z-component
        rational = aaa(sol.t, sol.y[4], mmax=mmax, tol=1e-14)
        pole, res = rational.polres(use_mp=False)

        A = 0j
    
        for p in range(len(pole)):
            # Need to cut out spurious poles where the real part of the
            # location lies outside of the interval of approximation
            if pole[p].imag >= 0 and \
               pole[p].real >= sol.t[0] and \
               pole[p].real <= sol.t[-1]:
                A += res[p] * np.exp(1j*pole[p]/eps)

        B_sur[k] = 2*np.pi*A/eps
    
        sol = solve_ivp(F, [sol.t[-1], t_final_dummy], sol.y[:,-1],
                        events=[event_up], rtol= 1e-9, atol= 1e-9)

    return E_num, B_sur
    
def MSD(z, N0=None):

    N = len(z)
    if N0 is None:
        N0 = N//10

    msd = np.zeros(N0)
    for n in range(N0):
        msd[n] = np.mean(abs(z[n:N-N0+n]-z[:-N0])**2)

    return msd

def plot_parameters(r=0.8, n=1):
    
    fig_width_pt = 360.0  # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt = r*fig_width_pt # Figure should span fraction r of text
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean/n    # height in inches
    fig_size =  [fig_width,fig_height]

    rcParams.update({'axes.labelsize': 9,
                     'axes.titlesize': 9,
                     'legend.fontsize': 6,
                     'xtick.labelsize': 5,
                     'ytick.labelsize': 5,
                     'xtick.major.size': 2,
                     'ytick.major.size': 2,
                     'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5,
                     'text.usetex': True,
                     'font.family': 'serif',
                     'lines.linewidth': 0.3,
                     'lines.markeredgewidth': 0.3,
                     'lines.markersize': 3,
                     'figure.figsize': fig_size,
                     'text.latex.preamble': r'\usepackage{amsmath}'})
