import numpy as np
import sympy as sp
from sympy.solvers.solveset import linsolve

a0, a1, a2, a3= sp.symbols("a0 a1 a2 a3")
h = sp.Symbol("h")

eq1 = sp.Eq(a0 + a1 + a2 + a3, 0)
eq2 = sp.Eq(-2*a0*h - a1*h + a3*h, -1)
eq3 = sp.Eq(2*a0*h**2 + 0.5*a1*h**2 + 0.5*a3*h**2, 0)
eq4 = sp.Eq((-8/6)*a0*h**3 - (1/6)*a1*h**3 + (1/6)*a3*h**3, 0)

ans = sp.solve([eq1,eq2,eq3,eq4], (a0,a1,a2,a3))
print(ans)