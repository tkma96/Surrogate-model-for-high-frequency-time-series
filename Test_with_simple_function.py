#!/usr/bin/env python
# coding: utf-8

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from baryrat import aaa
from common_functions import *

Max_order=40
eps= 0.25

#Derivatives of g
t = sym.symbols('t')
g = 1/sym.cosh(t)
G = [g]
for i in range(Max_order):
    p = sym.diff(G[-1],t)
    G.append(sym.horner(p))
    
H = [sym.lambdify((t), d) for d in G]

def simcase(t,v):
    return np.r_[v[1]/eps,
                 (-v[0]+1/(np.cosh(t)))/eps]

def F0(t,n):
    terms = [eps**i * (-1)**(i//2) * H[i](t) for i in range(0,n+1,2)]
    return sum(terms)

def F1(t,n):
    terms = [eps**i * (-1)**(i//2) * H[i](t) for i in range(1,n+1,2)]
    return sum(terms)

def E(q,p,t,n):
    return (q-F0(t,n))**2 + (p-F1(t,n))**2

def w(q,p,t,n):
    return np.c_[q-F0(t,n),p-F1(t,n)]

#intgration time
j = 15

qp0 = [F0(-15,j),F1(-15,j)]

sim_sys = solve_ivp(simcase, [-15, 15], qp0, rtol=1e-9, atol=1e-9)

tt = np.linspace(-15, 15, 50)
# Plot in paper is at 50 sample points; modify to 100 sample points
# for some of the results quoted in the text

r_interpolation = aaa(tt, 1/(np.cosh(tt)),
                      tol=1e-14,mmax=9,return_errors=False)
poles, res = r_interpolation.polres(use_mp=False)

asymp = 2*np.pi/eps * np.exp(-np.pi/2/eps)
asymp_next_pair = 2*np.pi/eps * np.exp(-3*np.pi/2/eps)
numerical = np.sqrt(E(sim_sys.y[0,-1], sim_sys.y[1,-1], 15, j))
aaa_connection_1_pole = 2*np.pi/eps * np.abs(np.exp(1j*poles[6]/eps)*res[6])
aaa_connection_2_poles = 2*np.pi/eps * np.abs(np.exp(1j*poles[6]/eps)*res[6]+
                                              np.exp(1j*poles[4]/eps)*res[4])
aaa_connection_all_poles = 2*np.pi/eps * np.abs(np.exp(1j*poles[6]/eps)*res[6]+
                                                np.exp(1j*poles[4]/eps)*res[4]+
                                                np.exp(1j*poles[2]/eps)*res[2]+
                                                np.exp(1j*poles[0]/eps)*res[0])
aaa_next_pair = 2*np.pi/eps * np.abs(np.exp(1j*poles[4]/eps)*res[4])

rel_errror_2_exact_poles = 100*(asymp-asymp_next_pair-numerical)/numerical
rel_errror_1_exact_poles = 100*(asymp-numerical)/numerical
rel_error_aaa_1_pole = 100*np.abs(aaa_connection_1_pole-numerical)/numerical
rel_error_aaa_2_poles = 100*np.abs(aaa_connection_2_poles-numerical)/numerical
rel_error_aaa_all_poles = 100*np.abs(aaa_connection_all_poles-numerical)/numerical
rel_contribution_next_pole = 100*asymp_next_pair/asymp

print("Poles:", poles)
print("Residues:", res)
#print("Asymptotic prediction of fast amplitude:", asymp)
#print("Numerical evaluation of fast amplitude:", numerical)
print(f"Relative error, 1 exact pair of poles: {rel_errror_1_exact_poles:7.5f}")
print(f"Relative error, 2 exact pairs of poles: {rel_errror_2_exact_poles:7.5f}")
print(f"Relative error AAA with 1 pair of poles: {rel_error_aaa_1_pole:7.5f}")
print(f"Relative error AAA with 2 pairs of poles: {rel_error_aaa_2_poles:7.5f}")
print(f"Relative error AAA with all pairs of poles: {rel_error_aaa_all_poles:7.5f}")
#print(f"Relative contribution of next pair of poles: {rel_contribution_next_pole:7.5f}")

xext = 3
yext = 3*np.pi
x = np.linspace(-xext, xext, 100)
y = np.linspace(-yext, yext, 100)
xx, yy = np.meshgrid(x, y)
zz = 1/np.cosh(xx + 1j*yy)

plot_parameters()

# For aspect ratio trick, see https://stackoverflow.com/a/7968690
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

plt.figure()
theta = 0.5*np.pi*np.arange(-5, 6)
plt.yticks(theta, [r'$-\tfrac52 \pi$', r'$-2\pi$', r'$-\tfrac32 \pi$',
                   r'$-\pi$', r'$-\tfrac12 \pi$', r'$0$', 
                   r'$\tfrac12 \pi$', r'$\pi$', r'$\tfrac32 \pi$', 
                   r'$2\pi$', r'$\tfrac52 \pi$'])
plt.scatter(poles.real,poles.imag,
            marker='x', c='k',
            s=40*np.abs(res))
plt.imshow(abs(zz), origin='lower', cmap = 'viridis',
           extent = (-xext, xext, -yext, yext),
           norm = plt.matplotlib.colors.LogNorm(vmin=0.01))
forceAspect(plt.gca())
plt.grid()
#plt.title(r"AAA-Estimate of poles of $\operatorname{sech}(z)$")
plt.xlabel(r"$\operatorname{Re}(t)$")
plt.ylabel(r"$\operatorname{Im}(t)$")
plt.savefig("aaa-sech.pdf", bbox_inches='tight')

### The following figures are not included in paper, but may be useful
### for presentations or for a paper.  Current version of code is not
### visually well-tuned

plt.figure()
plt.plot(sim_sys.t,1/(np.cosh(sim_sys.t)), 
         label = r'$g(t) = \operatorname{sech}(t)$', lw=3)
plt.plot(sim_sys.t, sim_sys.y[0],
         label = r'$q(t)$', lw=2.5)
plt.legend(prop={'size': 27})
plt.tick_params(labelsize=20)
#plt.savefig('solution-sech.pdf', bbox_inches='tight')

plt.figure()
first = len(sim_sys.t)//8
last = len(sim_sys.t)//2
for j in [3, 7, 9,10]:
    fast_energy = [E(sim_sys.y[0,k], sim_sys.y[1,k], sim_sys.t[k], j)
                   for k in range(first, last)]
    plt.plot(sim_sys.t[first:last], fast_energy,
             label=r'$E_{\text{fast}}^{'+str(j)+'}$', lw=3)
plt.legend(prop={'size': 30})
#plt.title(r'$\varepsilon = 0.25$, Varying order')
plt.xlabel(r"$t$",fontsize=30, color = 'black')
plt.ylabel(r"Fast Energy",fontsize=30, color = 'black')
plt.tick_params(labelsize=20)
#plt.savefig("fast_energy.pdf", bbox_inches='tight')

plt.show()

plt.figure()
for eps, j in [(0.25, 7), (0.1, 16), (0.05, 35)]:
    qp0 = [F0(-15,j),F1(-15,j)]
    tt = np.linspace(-2, 2, 100)
    sim_sys = solve_ivp(simcase, [-15, 15], qp0, rtol=1e-15, atol=1e-15,
                        t_eval = tt)
    fast_energy = [E(sim_sys.y[0,k], sim_sys.y[1,k], sim_sys.t[k], j)
                   for k in range(len(tt))]
    plt.plot(tt, fast_energy/fast_energy[-1],
             label=r'$\epsilon = '+str(eps)+'$, $N='+str(j)+'$', lw=3)
plt.legend(prop={'size': 25})
#plt.title(r'Optimal Truncation for different $\varepsilon$')
plt.xlabel(r"$t$",fontsize=30, color = 'black')
plt.ylabel(r"Relative Fast Energy",fontsize=30, color = 'black')
plt.tick_params(labelsize=20)
#plt.savefig("fast_energy_2.pdf", bbox_inches='tight')

plt.show()
