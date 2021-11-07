import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

n, h, x = sp.symbols("n h x")

steps = 100
h_min = 10**(-4) #min step size
h_max = 10**0 # max step size
fn = sp.sin(5*x) #fn being evaluated
ua = -25 * sp.sin(5*x) #exact 2nd derivative of fn
n_val = 1.5 #x value the fn is evaluated at

h_vals = np.linspace(h_min, h_max, num=steps)

u1 = (fn.subs(x, n + 2*h) - 2*fn.subs(x, n) + fn.subs(x, n - 2*h)) / (4*h**2) #derived formula
u2 = (fn.subs(x, n + h) - 2*fn.subs(x, n) + fn.subs(x, n - h)) / (h**2) #popular formula
ua_ans = ua.subs(x, n_val)

ans_1_array = np.zeros(steps)
ans_2_array = np.zeros(steps)
u1_err_array = np.zeros(steps)
u2_err_array = np.zeros(steps)

for i in range(len(h_vals)):
    ans_1 = u1.subs([(n, n_val), (h, h_vals[i])])
    ans_2 = u2.subs([(n, n_val), (h, h_vals[i])])
    
    u1_err = ua_ans - ans_1
    u2_err = ua_ans - ans_2

    ans_1_array[i] = ans_1
    ans_2_array[i] = ans_2
    u1_err_array[i] = u1_err
    u2_err_array[i] = u2_err

fig, ax = plt.subplots()
ax.loglog(h_vals, np.absolute(u1_err_array), label='derived formula')
ax.loglog(h_vals, np.absolute(u2_err_array), label='popular formula')
ax.loglog(h_vals, h_vals**2, label='second order accuracy')
ax.set_xlabel('step size')
ax.set_ylabel('absolute error')
ax.legend()
plt.show()
