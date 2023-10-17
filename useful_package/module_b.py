#!/usr/bin/python3


import numpy as np 


def hyperbola(x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    return 4*x**3 + 3*x**2 + 2*x + 1

