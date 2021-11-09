import numpy as np
import ipdb
from numba import jit

@jit
def SOR(phi, w, x_int_nodes, y_int_nodes, omega, dx, dy, max_error=1e-6):
    SOR_error = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    phi_temp = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    phi_temp[:,:] = phi[:,:]

    s = 1 / (2/dx**2 + 2/dy**2)
    num_bad_nodes = 1
    iter = 0
    
    while num_bad_nodes > 0:
        num_bad_nodes = 0

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                phi_temp[l,j] = ((1-omega) * phi[l,j] + omega * (s * ((1/dx**2) * (phi_temp[l-1,j] + phi[l+1,j]) + 
                                (1/dy**2) * (phi_temp[l,j-1] + phi[l,j+1]) - w[l,j])))

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                SOR_error[l,j] = np.abs(phi_temp[l,j] - phi[l,j])
                if SOR_error[l,j] >= max_error:
                    num_bad_nodes += 1

        iter += 1
        phi[:,:] = phi_temp[:,:]
    print(iter)

    return phi

@jit
def GS(phi, w, x_int_nodes, y_int_nodes, dx, max_error=1e-6):
    GS_error = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    phi_temp = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    phi_temp[:,:] = phi[:,:]

    num_bad_nodes = 1
    iter = 0

    while num_bad_nodes > 0:
        num_bad_nodes = 0

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                phi_temp[l,j] = (phi_temp[l-1,j] + phi_temp[l+1,j] + phi_temp[l,j-1] + phi_temp[l,j+1])/4 + (dx**2)*w[l,j]/4

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                GS_error[l,j] = np.abs(phi_temp[l,j] - phi[l,j])
                if GS_error[l,j] >= max_error:
                    num_bad_nodes += 1
        iter += 1
        phi[:,:] = phi_temp[:,:]
    # print(iter)

    return phi

@jit
def solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q, omega, nu, U):
    x_int_nodes = int(Lx / dx) - 1
    y_int_nodes = int(Ly / dy) - 1

    t_steps = int(t_f / dt)

    phi = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # streamfunction
    u = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # x velocity
    v = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # y velocity
    w = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # vorticity

    print(u.shape)
    u[:,y_int_nodes+1,:] = U # top surface moving plate BC

    # initialize omega, first order centered differences
    for l in range(1, x_int_nodes+1):
        for j in range(1, y_int_nodes+1):
            w[l,j,0] = (v[l+1,j,0] - v[l-1,j,0]) / (2*dx) - (u[l,j+1,0] - u[l,j-1,0]) / (2*dy)

    phi[:,:,0] = GS(phi[:,:,0], w[:,:,0], x_int_nodes, y_int_nodes, dx)
    #phi[:,:,0] = SOR(phi[:,:,0], w[:,:,0], x_int_nodes, y_int_nodes, omega, dx, dy)

    for n in range(t_steps-1):

        # omega boundary conditions

        # left and right boundaries
        for j in range(y_int_nodes+2):
            w[0,j,n] = -2 * (phi[1,j,n] - phi[0,j,n]) / dx**2 - 2 * v[0,j,n] / dx # left wall
            w[x_int_nodes+1,j,n] = -2 * (phi[x_int_nodes+1,j,n] - phi[x_int_nodes,j,n]) / dx**2 - 2 * v[x_int_nodes+1,j,n] / dx # right wall
        
        # top and bottom boundaries
        for l in range(x_int_nodes+2):
            w[l,0,n] = -2 * (phi[l,1,n] - phi[l,0,n]) / dy**2 + 2 * u[l,0,n] / dy # bottom wall
            w[l,y_int_nodes+1,n] = -2 * (phi[l,y_int_nodes+1,n] - phi[l,y_int_nodes,n]) / dy**2 + 2 * u[l,y_int_nodes+1,n] / dy # top wall
        
        # first solve for omega_n+1 without blending
        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                w[l,j,n+1] = (w[l,j,n] - (u[l,j,n]*dt/(2*dx)) * (w[l+1,j,n] - w[l-1,j,n]) - (v[l,j,n]*dt/(2*dy)) * (w[l,j+1,n] - w[l,j-1,n])
                              + (nu*dt/dx**2) * (w[l-1,j,n] - 2*w[l,j,n] + w[l+1,j,n]) + (nu*dt/dy**2) * (w[l,j-1,n] - 2*w[l,j,n] + w[l,j+1,n]))
        
                if q > 0: # blended discretization
                    if l > 1 and u[l,j,n] > 0: # 3rd order upwind u > 0
                        wx_minus = (w[l-2,j,n] - 3*w[l-1,j,n] + 3*w[l,j,n] - w[l+1,j,n]) / (3*dx)
                        w[l,j,n+1] -= dt*q*u[l,j,n]*wx_minus
                    elif l < x_int_nodes and u[l,j,n] < 0: # 3rd order upwind u < 0
                        wx_plus = (w[l-1,j,n] - 3*w[l,j,n] + 3*w[l+1,j,n] - w[l+2,j,n]) / (3*dx)
                        w[l,j,n+1] -= dt*q*u[l,j,n]*wx_plus

                    if j > 1 and v[l,j,n] > 0: # 3rd order upwind v > 0
                        wy_minus = (w[l,j-2,n] - 3*w[l,j-1,n] + 3*w[l,j,n] - w[l,j+1,n]) / (3*dy)
                        w[l,j,n+1] -= dt*q*v[l,j,n]*wy_minus

                    elif j < y_int_nodes and v[l,j,n] < 0: # 3rd order upwind v < 0
                        wy_plus = (w[l,j-1,n] - 3*w[l,j,n] + 3*w[l,j+1,n] - w[l,j+2,n]) / (3*dy)
                        w[l,j,n+1] -= dt*q*v[l,j,n]*wy_plus

        phi[:,:,n+1] = GS(phi[:,:,n], w[:,:,n+1], x_int_nodes, y_int_nodes, dx)
        #phi[:,:,n+1] = SOR(phi[:,:,n], w[:,:,n+1], x_int_nodes, y_int_nodes, omega, dx, dy)
        # if(n > 100):
        #     ipdb.set_trace()
        print(n+1)
        

        
    return 'ok'