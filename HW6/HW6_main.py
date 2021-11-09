import numpy as np
import matplotlib.pyplot as plt

import HW6_Q1 as Q1


def main():
    # domain params
    Lx = 0.4 # m
    Ly = 0.4 # m

    # scheme params
    dx = 0.01 # m
    dy = 0.01 # m
    dt = 0.002 # s
    t_sol = (0.5, 1.5, 3.0) # times of interest, s
    t_f = 3.0 # final time, s
    q = 0.5 # for blended discretization
    omega = 1.8 # for streamfunction SOR iteration


    # physics params
    U = 5 # m/s
    rho = 1 # kg/m^3
    Re = 200
    nu = 1 / (Re / (U * Lx)) # m^2 / s

    result = Q1.solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q, omega, nu, U)

    x_grid = np.arange(0, Lx+dx, dx)
    y_grid = np.arange(0, Ly+dy, dy)
    u = result[0]
    v = result[1]
    plt.quiver(x_grid, y_grid, u[:,:,20].T, v[:,:,20].T)
    plt.show()





if __name__ == '__main__':
    main()