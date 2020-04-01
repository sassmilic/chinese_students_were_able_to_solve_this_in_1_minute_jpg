#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve, minimize
import scipy.optimize as opt

VARS = ['a1', 'a2', 'b1', 'b2', 'h', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'h1', 'h2']
V = {v:i for i,v in enumerate(VARS)}

def main():
    n = len(VARS)
    Z = np.empty((n))
    zGuess = np.ones((n))
    # constraints: all variables must be greater than 0
    cons = {'type' : 'ineq', 'fun': constraints}
    res = minimize(f, zGuess, method='SLSQP', constraints=cons, options={'ftol': 1e-10})
    
    for v,x in list(zip(VARS, res.x)):
        print(v, round(x, 2))
   
def f(Z):
    a1 = Z[V['a1']]
    a2 = Z[V['a2']]
    b1 = Z[V['b1']]
    b2 = Z[V['b2']]
    h = Z[V['h']]
    t = Z[V['t']]
    u = Z[V['u']]
    v = Z[V['v']]
    w = Z[V['w']]
    x = Z[V['w']]
    y = Z[V['y']]
    z = Z[V['z']]
    h1 = Z[V['h1']]
    h2 = Z[V['h2']]

    triangle = lambda b, h : b * h / 2
    pgram = lambda b, h : b * h
    trapezoid = lambda b1, b2, h : (b1 + b2) * h / 2

    F = np.empty(Z.shape)

    # triangles
    F[0] = triangle(b1, h) - (79 + v + x)
    F[1] = triangle(b2, h) - (10 + y + z)
    F[2] = triangle(a1, h) - (u + t)
    F[3] = triangle(a2, h) - (72 + 8 + w)
    F[4] = triangle(a1 + a2, h1) - (72 + u + v + y)
    F[5] = triangle(b1 + b2, h2) - (8 + x + z)
    
    # trapezoids
    F[6] = trapezoid(a1, b1, h) - (79 + t + u + v + x)
    F[7] = trapezoid(a2, b1, h) - (72 + 79 + 8 + v + w + x)
    F[8] = trapezoid(a2, b2, h) - (72 + 10 + 8 + w + y + z)
    F[9] = trapezoid(a1 + a2, b1, h) - (79 + 72 + 8 + t + u + v + w + x)
    F[10] = trapezoid(a2, b1 + b2, h) - (79 + 72 + 8 + 10 + v + x + w + y + z)
    
    # parallelogram
    F[11] = pgram(a1 + a2, h) - (79 + 72 + 8 + 10 + t + u + v + w + x + y + z)
    
    # a1 + a2 == b1 + b2
    F[12] = (a1 + a2) - (b1 + b2)
    # h1 + h2 == h
    F[13] = h - (h1 + h2)

    return np.dot(F, F)

def constraints(z):
    f = np.zeros_like(z)
    for i,_ in enumerate(z):
        f[i] = z[i]
    return f


if __name__ == "__main__":
    main()

