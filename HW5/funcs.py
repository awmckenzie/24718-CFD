import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import ipdb

def create_GIF(x_grid, y_grid, sol, dt):
    filenames = []
    for i in range(len(sol)):
        plt.contourf(x_grid, y_grid, sol[i], levels=50)
        # plt.xlabel('Length (m)')
        # plt.ylabel('H + ' + r'$\eta$' + ' (m)')
        # plt.title('Lax-W with Periodic BCs, CFL = 1.0')
        # plt.grid()
        # plt.legend(loc=2)
        filename = f'{i}.png'
        filenames.append(filename)

        plt.savefig(filename)
        plt.close()

    # build gif
    with io.get_writer('soln.gif', mode='I', duration=dt*50) as writer:
        for filename in filenames:
            image = io.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

def error_abs_rms(T_exact, T_scheme, Nx, Ny):
    err = 0
    for l in range(Nx):
        for j in range(Ny):
            err += np.abs(T_scheme[l,j] - T_exact[l,j])**2
    err = np.sqrt(err/(Nx*Ny))
    return err

def error_rel_rms(T_exact, T_scheme, Nx, Ny):
    err = 0
    for l in range(1, Nx-1):
        for j in range(1, Ny-1):
            err += np.abs((T_scheme[l,j] - T_exact[l,j])/T_exact[l,j])**2
    err = np.sqrt(err/(Nx*Ny))
    return err

def error_abs(T_exact, T_scheme, Nx, Ny):
    err = 0
    for l in range(Nx):
        for j in range(Ny):
            err += T_scheme[l,j] - T_exact[l,j]
    err = err/(Nx*Ny)
    return err

def get_points(T, pts, Lx, Ly, Nx, Ny):
    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)
    vals = []
    for pt in pts:
        x_ind = int(pt[0]/dx)
        y_ind = int(pt[1]/dy)
        vals.append(T[x_ind, y_ind])

    return vals
