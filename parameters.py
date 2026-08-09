#!/usr/bin/env python
# coding: utf-8

# The parameter b is always kept fixed at its classical value
b = 8/3

# Classical Lorenz parameters (chaotic regime)
chaotic = {'r': 28,
           'sigma': 10,
           'b': b,
           'eps': 0.01,
           'order': 16,
           'threshold': 30,
           'mmax': 7,
           'y0': [0, 0, -14, -15, 20]}

# Dullin, Schmidt, Richter, Grossmann, 2006; their point A1
periodic = {'r': 402,
            'sigma': 302,
            'b': b,
            'eps': 0.001,
            'order': 11,
            'threshold': 390,
            'mmax': 11,
            'y0': [0, 0, 0.001, 0.001, 401]}

# Period 2 orbit in the red region of
# Dullin, Schmidt, Richter, Grossmann, 2006, Figure 1
# Note: orders substantially larger than 8 produce overshoots
periodic2 = {'r': 120,
             'sigma': 5.5,
             'b': b,
             'eps': 0.01,
             'order': 8,
             'threshold': 98.0,
             'mmax': 9,
             'y0': [0, 0, 0.001, 0.001, 119]}

