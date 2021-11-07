import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import full

import Q1_Part1 as Q1P1
import Q1_Part2 as Q1P2
import funcs as fn

import ipdb

def main():
    
    # scheme params
    dx = 0.01 # m
    dy = 0.01 # m
    Lx = 3 # m
    Ly = 3 # m

    t = 4 # s
    dt_1 = 0.0002
    dt_2 = 0.002

    # advection params
    u = 1.5 # m/s
    v_1 = 1.0 # m/s
    v_2 = -0.5 # m/s

    # diffusion params
    mu_1 = 0.1 # m^2/s
    mu_2 = 0.001 # m^2/s

    # initial condition
    phi_0 = 0

    # boundary conditions
    phi_0y = (0, 1, 0) # (0 < x < Lx/3), (Lx/3 < x < 2Lx/3), (2Lx/3 < x < Lx)

    dphi_x0 = 0
    dphi_xLy = 0
    dphi_Lxy = 0

    x_grid = np.arange(0, Lx+dx, dx)
    y_grid = np.arange(0, Ly+dy, dy)

    sol_1 = Q1P1.scheme(dx, dy, Lx, Ly, u, v_1, mu_1, t, dt_1, phi_0y, dphi_x0, dphi_xLy, dphi_Lxy, phi_0)
    result_1 = sol_1[0][int(t/sol_1[1])-1,:,:]
    fig1, ax1 = plt.subplots()
    im1 = ax1.contourf(x_grid, y_grid, result_1.T, levels=50)
    ax1.set_title('Pollutant Density, ' + r'$\mu$' + ' = ' + str(mu_1) + 'm^2/s')
    ax1.set_ylabel('y (m)')

    sol_2 = Q1P1.scheme(dx, dy, Lx, Ly, u, v_1, mu_2, t, dt_2, phi_0y, dphi_x0, dphi_xLy, dphi_Lxy, phi_0)
    result_2 = sol_2[0][int(t/sol_2[1])-1,:,:]
    fig2, ax2 = plt.subplots()
    im2 = ax2.contourf(x_grid, y_grid, result_2.T, levels=50)
    ax2.set_title('Pollutant Density, ' + r'$\mu$' + ' = ' + str(mu_2) + 'm^2/s')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

    sol_3 = Q1P2.scheme(dx, dy, Lx, Ly, u, v_2, mu_1, t, dt_1, dphi_x0, dphi_xLy, dphi_Lxy, phi_0y, result_1)
    result_3 = sol_3[0][int(t/sol_3[1])-1,:,:]
    fig3, ax3 = plt.subplots()
    im3 = ax3.contourf(x_grid, y_grid, result_3.T, levels=50)
    ax3.set_title('Pollutant Density, ' + r'$\mu$' + ' = ' + str(mu_1) + 'm^2/s')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')

    sol_4 = Q1P2.scheme(dx, dy, Lx, Ly, u, v_2, mu_2, t, dt_2, dphi_x0, dphi_xLy, dphi_Lxy, phi_0y, result_2)
    result_4 = sol_4[0][int(t/sol_4[1])-1,:,:]
    fig4, ax4 = plt.subplots()
    im4 = ax4.contourf(x_grid, y_grid, result_4.T, levels=50)
    ax4.set_title('Pollutant Density, ' + r'$\mu$' + ' = ' + str(mu_2) + 'm^2/s')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')

    plt.show()

    # animation code for fun
    # full_result_1 = []
    # full_result_2 = []
    # count = 0
    # for i in range(len(sol_1[0])):
    #     if i%50 == 0:
    #         full_result_1.append(sol_1[0][i,:,:])
    #         full_result_2.append(sol_3[0][i,:,:])
    #         count += 1
    # full_result = full_result_1 + full_result_2
    # fn.create_GIF(x_grid, y_grid, full_result, sol_1[1])

if __name__ == '__main__':
    main()