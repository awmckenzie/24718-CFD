import Q2_exact as Q2e
import Q2_Point_Jacobi as PJ
import Q2_Gauss_Seidel as GS
import Q2_SOR as SOR

import funcs as fn
import time as t
import matplotlib.pyplot as plt
import matplotlib.colors as c

def main():

    # scheme params
    Lx = 2 # m
    Ly = 1 # m

    Nx = 21
    Ny = 21

    T_exact = Q2e.exact(Lx, Ly, Nx, Ny)
    T_PJ = PJ.PJ(Lx, Ly, Nx, Ny)
    T_GS = GS.GS(Lx, Ly, Nx, Ny)
    T_SOR = SOR.SOR(Lx, Ly, Nx, Ny)
    contour_levels = 16

    print(fn.error_rel_rms(T_exact[0], T_PJ[0], Nx, Ny))
    print(fn.error_abs_rms(T_exact[0], T_PJ[0], Nx, Ny))

    print(fn.error_rel_rms(T_exact[0], T_GS[0], Nx, Ny))
    print(fn.error_abs_rms(T_exact[0], T_GS[0], Nx, Ny))

    print(fn.error_rel_rms(T_exact[0], T_SOR[0], Nx, Ny))
    print(fn.error_abs_rms(T_exact[0], T_SOR[0], Nx, Ny))

    pts = [(0.5,0.25),(0.5,0.75),(1.5,0.25),(1.5,0.75)]
    print(fn.get_points(T_PJ[0], pts, Lx, Ly, Nx, Ny))
    print(fn.get_points(T_GS[0], pts, Lx, Ly, Nx, Ny))
    print(fn.get_points(T_SOR[0], pts, Lx, Ly, Nx, Ny))
    print(fn.get_points(T_exact[0], pts, Lx, Ly, Nx, Ny))

    fig1, ax1 = plt.subplots()
    im1 = ax1.contour(T_exact[1], T_exact[2], T_exact[0], levels=contour_levels, linestyles='dashed')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    norm= c.Normalize(vmin=im1.cvalues.min(), vmax=im1.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = im1.cmap)
    sm.set_array([])
    cbar1 = plt.colorbar(sm, location='right', ticks=im1.levels)
    cbar1.set_label('Temperature (C)')
    im2 = ax1.contour(T_PJ[1], T_PJ[2], T_PJ[0], levels=contour_levels)
    ax1.set_title('Point-Jacobi Method')

    fig2, ax2 = plt.subplots()
    im3 = ax2.contour(T_exact[1], T_exact[2], T_exact[0], levels=contour_levels, linestyles='dashed')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    norm= c.Normalize(vmin=im3.cvalues.min(), vmax=im3.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = im3.cmap)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, location='right', ticks=im3.levels)
    cbar2.set_label('Temperature (C)')
    im4 = ax2.contour(T_GS[1], T_GS[2], T_GS[0], levels=contour_levels)
    ax2.set_title('Gauss-Seidel Method')

    fig3, ax3 = plt.subplots()
    im5 = ax3.contour(T_exact[1], T_exact[2], T_exact[0], levels=contour_levels, linestyles='dashed')
    im6 = ax3.contour(T_SOR[1], T_SOR[2], T_SOR[0], levels=contour_levels)
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    norm= c.Normalize(vmin=im6.cvalues.min(), vmax=im6.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = im6.cmap)
    sm.set_array([])
    cbar3 = plt.colorbar(sm, location='right', ticks=im6.levels)
    cbar3.set_label('Temperature (C)')
    ax3.set_title('Gauss-Seidel Method with SOR')

    plt.show()

if __name__ == '__main__':
    main()