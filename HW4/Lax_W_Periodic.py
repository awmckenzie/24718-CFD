import numpy as np

def LAXW(x1, x2, dx, Lx, t, CFL, n_0, n_Lx, U, A, lam):

    dt = CFL * dx / U
    t_steps = int(t/dt)
    x_steps = int(Lx/dx)
    x_range = np.linspace(0, Lx, x_steps)
    eta = np.zeros(x_steps) # IC

    for i in range(x_steps):
        if x_range[i] < x1 or x_range[i] > x2:
            eta[i] = 0
        else:
            eta[i] = A * np.sin( 2 * np.pi * x_range[i] / lam)

    res = np.empty([t_steps, x_steps]) # result matrix
    res[0] = eta # set IC
    res[0, 0] = 0 # BC (0,0) 

    for n in range(t_steps-1):
        res[n+1, 1:x_steps-1] = (res[n, 1:x_steps-1] - (CFL/2) * (res[n, 2:x_steps] - res[n, 0:x_steps-2])
                                + (CFL**2)/2 * (res[n, 2:x_steps] - 2*res[n, 1:x_steps-1] + res[n, 0:x_steps-2]))
        res[n+1, 0] = res[n+1, x_steps-2]
        res[n+1, x_steps-1] = res[n+1, 1]

    return x_range, res, dt
