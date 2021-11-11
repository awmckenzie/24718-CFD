import numpy as np
import matplotlib.pyplot as plt
import ipdb

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
    omega = 1.8972 # for streamfunction SOR iteration

    # physics params
    U = 5 # m/s
    u_in = 1.2 # inlet outlet, m/s
    rho = 1 # kg/m^3
    Re = 200
    nu = 1 / (Re / (U * Lx)) # m^2 / s

    result = Q1.solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q, omega, nu, U, u_in)

    x_grid = np.arange(0, Lx+dx, dx)
    y_grid = np.arange(0, Ly+dy, dy)
    u = result[0]
    v = result[1]
    fig1 = plt.figure()
    plt.quiver(x_grid, y_grid, u[:,:,int(t_f/dt)-1].T, v[:,:,int(t_f/dt)-1].T)
    fig2 = plt.figure()
    plt.streamplot(x_grid, y_grid, u[:,:,int(t_f/dt)-1].T, v[:,:,int(t_f/dt)-1].T, density=4/3, linewidth=1)
    plt.show()

if __name__ == '__main__':
    main()