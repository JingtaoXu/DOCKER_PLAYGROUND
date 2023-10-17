#!/usr/bin/python3

import numpy as np

def polynom_3(coefs, x):
    return np.sum([coefs[i]*x**(len(coefs)-i-1) for i in range(len(coefs))],axis=0)
