import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import interpolate as ip

import Crank_Nicolson_2D as CN2D
import ADI_2D_Dirichlet as ADI2DD
import ADI_2D_Neumann as ADI2DN

def main():

    #scheme params
    dx = 0.01 # x step, ft
    dy = 0.01 # y step, ft
    dt = 0.001 # time step, hours

    Lx = 0.32 # x length, ft
    Ly = 0.12 # y length, ft
    t = 0.5 # final time, hours

    #thermal params
    k = 0.004 # thermal condutivity, W/ftK
    Cp = 37.1 # specific heat, J/lbK
    rho = 9 # density, lb/ft^3

    #Dirichlet BCs
    T_0y = 300 # x=0, degK
    T_Lxy = 300 # x=Lx, degK
    T_x0 = 0 # y=0, degK
    T_xLy = 1650
    T_xLy_arr = np.zeros([int(t/dt)])
    T_xLy_arr.fill(1650) # y=Ly, degK, this will be a excel sheet later
    
    t_ree = np.array([0  , 250 , 300 , 350 , 400 , 500 , 600 , 800 , 1000, 1250, 1500, 1900, 2000])
    t_ree = np.divide(t_ree, 3600)
    T_ree = [300, 1200, 1300, 1600, 1700, 1800, 1700, 1700, 1600, 1200, 500 , 300 , 300]
    f = ip.interp1d(t_ree, T_ree)
    t_ree_new = np.arange(0,t,dt)
    T_ree_new = f(t_ree_new)
    print(T_ree_new)

    #Neumann BCs
    dT_0y = -900 / k # x=0, degK/ft
    dT_Lxy = 900 / k # x=Lx, degK/ft
    dT_x0 = 0 # y=0, degK/ft

    #IC
    T_0 = 300 # t=0, degK

    a = 3600*(k/(rho*Cp)) # thermal diffusivity, ft^2/h
    time1 = time.time()
    result_CN = CN2D.CN(dx, dy, dt, Lx, Ly, t, T_0y, T_Lxy, T_x0, T_xLy, T_0, a)
    time2 = time.time()
    result_ADI_D = ADI2DD.ADI(dx, dy, dt, Lx, Ly, t, T_0y, T_Lxy, T_x0, T_xLy_arr, T_0, a)
    time3 = time.time()
    ADI2DD.ADI(dx, dy, dt, Lx, Ly, t, T_0y, T_Lxy, T_x0, T_xLy_arr, T_0, a, True)
    time4 = time.time()
    result_ADI_TDMA_N = ADI2DN.ADI(dx, dy, dt, Lx, Ly, t, dT_0y, dT_Lxy, dT_x0, T_xLy, T_0, a, True)

    result_ADI_TDMA_D2 = ADI2DD.ADI(dx, dy, dt, Lx, Ly, t, T_0y, T_Lxy, T_x0, T_ree_new, T_0, a)

    CN_time = time2 - time1
    ADI_time = time3 - time2
    ADI_TDMA_time = time4 - time3

    x_plot = np.arange(0,Lx+dx,dx)
    y_plot = np.arange(0,Ly+dy,dy)
    t_plot = np.arange(dt,t,dt)

    fig1, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.contourf(x_plot, y_plot, result_CN, levels=50, cmap='plasma')
    im2 = ax2.contourf(x_plot, y_plot, result_ADI_D, levels=50, cmap='plasma')
    
    ax1.set_title('Crank-Nicolson, t=0.5h')
    ax2.set_title('Alternating Direction Implicit, t=0.5h')
    ax1.set_xlabel('x (ft)')
    ax2.set_xlabel('x (ft)')
    ax1.set_ylabel('y (ft)')
    ax2.set_ylabel('y (ft)')
    cbar1 = plt.colorbar(im1, location='bottom', ax=ax1)
    cbar2 = plt.colorbar(im2, location='bottom', ax=ax2)
    cbar1.set_label("Temperature (K)")
    cbar2.set_label("Temperature (K)")

    fig2, ax3 = plt.subplots()
    methods = ['Crank-Nicolson', 'ADI', 'ADI_TDMA']
    times = [CN_time, ADI_time, ADI_TDMA_time]
    ax3.bar(methods, times)
    ax3.set_title("Comparison of Methods CPU time")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("method")

    fig3, ax4 = plt.subplots()
    im3 = ax4.contourf(x_plot, y_plot, result_ADI_TDMA_N[int(t/dt)-1], levels=50, cmap='plasma')
    ax4.set_title("ADI Neumann with Input Heat Flux, t=0.05h")
    ax4.set_xlabel('x (ft)')
    ax4.set_ylabel('y (ft)')
    cbar3 = plt.colorbar(im3, location='bottom', ax=ax4)
    cbar3.set_label('Temperature (K)')

    y_temps = []
    for i in range(int(t/dt)-1):
        res = result_ADI_TDMA_N[i]
        y_temps.append(res[:, int(0.16/dx)])

    fig4, ax5 = plt.subplots()
    ax5.set_title("ADI Neumann with Input Heat Flux")
    ax5.set_xlabel('y (ft)')
    ax5.set_ylabel('t (s)')
    im4 = ax5.contourf(y_plot, t_plot, y_temps, levels=50, cmap='plasma')
    cbar9 = plt.colorbar(im4, location='bottom', ax=ax5)

    # fig5, ax7 = plt.subplots(1,3)

    # im6 = ax7.contourf(x_plot, y_plot, result_ADI_TDMA_D2, levels=50, cmap='plasma')  
    # ax7.set_title('t = 0.5h')
    # ax7.set_xlabel('x (ft)')   
    # ax7.set_ylabel('y (ft)')
    # cbar4 = plt.colorbar(im6, location='bottom', ax=ax7)
    # cbar4.set_label("Temperature (K)")

    plt.show()

if __name__ == '__main__':
    main()