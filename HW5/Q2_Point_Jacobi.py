import numpy as np
import ipdb

def PJ(Lx, Ly, Nx, Ny):

    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)
    s = 1 / (2/dx**2 + 2/dy**2)
    x_range = np.linspace(0, Lx, Nx)
    y_range = np.linspace(0, Ly, Ny)
    T = np.zeros((Nx, Ny))
    T_temp = np.zeros((Nx, Ny))
    iter = 0
    num_bad_nodes = 1
    max_err = 10**(-5)

    T[0,:] = 0 # left side BC
    for j in range(Ny):
        T[Nx-1, j] = j*dy # right side BC
    
    T_temp[:,:] = T[:,:]
    while num_bad_nodes > 0:

        error_abs = 0
        error_abs_rms = 0
        error_rel_rms = 0
        num_bad_nodes = 0

        for l in range(1, Nx-1):
            for j in range(1, Ny-1):
                # T_temp[l,j] = (T[l-1,j] + T[l+1,j] + (dx**2/dy**2)*(T[l,j+1] + T[l,j-1])) / (2 * (1 + (dx**2/dy**2)))
                T_temp[l,j] = s * ((1/dx**2) * (T[l-1,j] + T[l+1,j]) + (1/dy**2) * (T[l,j-1] + T[l,j+1]))

        # Neumann BCs
        T_temp[1:Nx-1,Ny-1] = T_temp[1:Nx-1,Ny-2] 
        T_temp[1:Nx-1,0] = T_temp[1:Nx-1,1]

        for l in range(1, Nx-1):
            for j in range(1, Ny-1):
                error = np.abs(T_temp[l,j] - T[l,j])
                if error >= max_err:
                    num_bad_nodes += 1
        
        # ipdb.set_trace()

        iter +=1
        T[:,:] = T_temp[:,:]
    print(iter)
    return T.T, x_range, y_range