import numpy as np
import matplotlib.pyplot as plt
import ipdb

import HW6_Q1 as Q1
import HW6_Q2 as Q2

# hello beepis

def main():
    # domain params
    Lx = 0.4 # m
    Ly = 0.4 # m

    # scheme params
    dx = 0.01 # m``
    dy = 0.01 # m
    dt = 0.002 # s
    t_sol = (0.5, 1.5, 3.0) # times of interest, s
    x_sol = 0.2 # x position of interest
    y_sol = 0.3 # y position of interest
    t_f = 3.0 # final time, s
    q1 = 0 # for blended discretization
    q2 = 0.5
    q3 = 0.25
    omega = 1.8972 # for streamfunction SOR iteration

    # physics params
    U = 5 # m/s
    u_in = 1.2 # inlet outlet, m/s
    rho = 1 # kg/m^3
    Re = 200
    nu = 1 / (Re / (U * Lx)) # m^2 / s

    result_Q11 = Q1.solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q1, omega, nu, U)
    result_Q12 = Q1.solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q2, omega, nu, U)
    result_Q2 = Q2.solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q3, omega, nu, U, u_in)

    

    x_grid = np.arange(0, Lx+dx, dx)
    y_grid = np.arange(0, Ly+dy, dy)

    u_Q11 = result_Q11[0]
    v_Q11 = result_Q11[1]
    res_Q11 = np.sqrt(np.square(u_Q11[:,:,int(t_f/dt)-1].T) + np.square(v_Q11[:,:,int(t_f/dt)-1].T))
    w_Q11 = result_Q11[2]

    u_Q12 = result_Q12[0]
    v_Q12 = result_Q12[1]
    res_Q12 = np.sqrt(np.square(u_Q12[:,:,int(t_f/dt)-1].T) + np.square(v_Q12[:,:,int(t_f/dt)-1].T))
    w_Q12 = result_Q12[2]


    u_Q2 = result_Q2[0]
    v_Q2 = result_Q2[1]
    res_Q2 = np.sqrt(np.square(u_Q2[:,:,int(t_f/dt)-1].T) + np.square(v_Q2[:,:,int(t_f/dt)-1].T))
    w_Q2 = result_Q2[2]

    x_c = int(x_sol/dx)-1
    y_c = int(y_sol/dy)
    t1 = int(t_sol[0]/dt) - 1
    t2 = int(t_sol[1]/dt) - 1
    t3 = int(t_sol[2]/dt) - 1

    # Q1 solution values
    print(u_Q11[x_c,y_c,t1], u_Q11[x_c,y_c,t2], u_Q11[x_c,y_c,t3])
    print(u_Q12[x_c,y_c,t1], u_Q12[x_c,y_c,t2], u_Q12[x_c,y_c,t3])

    print(v_Q11[x_c,y_c,t1], v_Q11[x_c,y_c,t2], v_Q11[x_c,y_c,t3])
    print(v_Q12[x_c,y_c,t1], v_Q12[x_c,y_c,t2], v_Q12[x_c,y_c,t3])

    print(w_Q11[x_c,y_c,t1], w_Q11[x_c,y_c,t2], w_Q11[x_c,y_c,t3])
    print(w_Q12[x_c,y_c,t1], w_Q12[x_c,y_c,t2], w_Q12[x_c,y_c,t3])

    # Q2 solution values
    print()
    print(u_Q2[x_c,y_c,t1], u_Q2[x_c,y_c,t2], u_Q2[x_c,y_c,t3])

    print(v_Q2[x_c,y_c,t1], v_Q2[x_c,y_c,t2], v_Q2[x_c,y_c,t3])

    print(w_Q2[x_c,y_c,t1], w_Q2[x_c,y_c,t2], w_Q2[x_c,y_c,t3])

    fig1, ax1 = plt.subplots()
    im1b = ax1.quiver(x_grid, y_grid, u_Q11[:,:,int(t_f/dt)-1].T, v_Q11[:,:,int(t_f/dt)-1].T)
    ax1.set_title('Velocity Vectors')
    ax1.set_ylabel('y (m)')
    ax1.set_xlabel('x (m)')

    fig2, ax2 = plt.subplots()
    im2a = ax2.contourf(x_grid, y_grid, res_Q11)
    im2b = ax2.streamplot(x_grid, y_grid, u_Q11[:,:,int(t_f/dt)-1].T, v_Q11[:,:,int(t_f/dt)-1].T, density=4/3, linewidth=1)
    cbar2 = plt.colorbar(im2a, location='right', ax=ax2)
    ax2.set_title('Velocity (m/s)')
    ax2.set_ylabel('y (m)')
    ax2.set_xlabel('x (m)')

    fig3, ax3 = plt.subplots()
    im3 = ax3.contourf(x_grid, y_grid, w_Q11[:,:,int(t_f/dt)-1].T)
    cbar3 = plt.colorbar(im3, location='right', ax=ax3)
    ax3.set_title('omega')
    ax3.set_ylabel('y (m)')
    ax3.set_xlabel('x (m)')

    fig4, ax4 = plt.subplots()
    im4b = ax4.quiver(x_grid, y_grid, u_Q12[:,:,int(t_f/dt)-1].T, v_Q12[:,:,int(t_f/dt)-1].T)
    ax4.set_title('Velocity Vectors')
    ax4.set_ylabel('y (m)')
    ax4.set_xlabel('x (m)')

    fig5, ax5 = plt.subplots()
    im5a = ax5.contourf(x_grid, y_grid, res_Q12)
    im5b = ax5.streamplot(x_grid, y_grid, u_Q12[:,:,int(t_f/dt)-1].T, v_Q12[:,:,int(t_f/dt)-1].T, density=4/3, linewidth=1)
    cbar5 = plt.colorbar(im5a, location='right', ax=ax5)
    ax5.set_title('Velocity (m/s)')
    ax5.set_ylabel('y (m)')
    ax5.set_xlabel('x (m)')

    fig6, ax6 = plt.subplots()
    im6a = ax6.contourf(x_grid, y_grid, w_Q12[:,:,int(t_f/dt)-1].T)
    cbar6 = plt.colorbar(im6a, location='right', ax=ax6)
    ax6.set_title('omega')
    ax6.set_ylabel('y (m)')
    ax6.set_xlabel('x (m)')

    fig7, ax7 = plt.subplots()
    im7b = ax7.quiver(x_grid, y_grid, u_Q2[:,:,int(t_f/dt)-1].T, v_Q2[:,:,int(t_f/dt)-1].T)
    ax7.set_title('Velocity Vectors')
    ax7.set_ylabel('y (m)')
    ax7.set_xlabel('x (m)')

    fig8, ax8 = plt.subplots()
    im8a = ax8.contourf(x_grid, y_grid, res_Q2)
    im8b = ax8.streamplot(x_grid, y_grid, u_Q2[:,:,int(t_f/dt)-1].T, v_Q2[:,:,int(t_f/dt)-1].T, density=4/3, linewidth=1)
    cbar8 = plt.colorbar(im8a, location='right', ax=ax8)
    ax8.set_title('Velocity (m/s)')
    ax8.set_ylabel('y (m)')
    ax8.set_xlabel('x (m)')

    fig9, ax9 = plt.subplots()
    im9a = ax9.contourf(x_grid, y_grid, w_Q2[:,:,int(t_f/dt)-1].T)
    cbar9 = plt.colorbar(im9a, location='right', ax=ax9)
    ax9.set_title('omega')
    ax9.set_ylabel('y (m)')
    ax9.set_xlabel('x (m)')

    plt.show()
if __name__ == '__main__':
    main()