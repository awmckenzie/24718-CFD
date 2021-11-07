import numpy as np

def exact(Lx, Ly, Nx, Ny):
    
    T = np.empty((Nx, Ny))
    x_range = np.linspace(0, Lx, Nx)
    y_range = np.linspace(0, Ly, Ny)

    for l in range(Nx):
        for j in range(Ny):
            val = 0
            for n in range(1, 103, 2):
                val += (1 / ((n * np.pi)**2 * np.sinh(2*n*np.pi))) * np.sinh(n*np.pi*x_range[l]) * np.cos(n*np.pi*y_range[j])
            T[l,j] = (x_range[l] / 4) - 4 * val

    return T.T, x_range, y_range