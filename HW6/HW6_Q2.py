from sys import _xoptions
import numpy as np
import ipdb
from numba import jit

@jit
def SOR(psi, w, u, v, x_int_nodes, y_int_nodes, omega, dx, dy, max_error=1e-6):

    s = 1 / (2/dx**2 + 2/dy**2)
    num_bad_nodes = 1
    iter = 0

    psi_temp = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    psi_temp[:,:] = psi[:,:]
    
    while num_bad_nodes > 0:
        num_bad_nodes = 0

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                psi_temp[l,j] = ((1-omega) * psi[l,j] + omega * (s * ((1/dx**2) * (psi_temp[l-1,j] + psi[l+1,j]) + 
                                (1/dy**2) * (psi_temp[l,j-1] + psi[l,j+1]) + w[l,j])))

        for j in range(int((y_int_nodes+2)/2),int(3*(y_int_nodes+2)/4)): # left wall inlet
            psi_temp[0,j] = (4*psi_temp[1,j] - psi_temp[2,j] + 2*dx*v[0,j]) / 3
    
        for j in range(int((y_int_nodes+2)/4),int((y_int_nodes+2)/2)): # right wall outlet
            psi_temp[x_int_nodes+1,j] = (4*psi_temp[x_int_nodes,j] - psi_temp[x_int_nodes-1,j] - 2*dx*v[x_int_nodes+1,j]) / 3

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                error= np.abs(psi_temp[l,j] - psi[l,j])
                if error >= max_error:
                    num_bad_nodes += 1

        iter += 1
        psi[:,:] = psi_temp[:,:]
    #print(iter)

    return psi

@jit
def GS(psi, w, u, v, x_int_nodes, y_int_nodes, dx, dy, max_error=1e-6):
    s = 1 / (2/dx**2 + 2/dy**2)
    num_bad_nodes = 1
    iter = 0
    psi_temp = np.zeros((x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    psi_temp[:,:] = psi[:,:]

    while num_bad_nodes > 0:
        num_bad_nodes = 0

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                psi_temp[l,j] = s * ((1/dx**2) * (psi_temp[l-1,j] + psi[l+1,j]) + (1/dy**2) * (psi_temp[l,j-1] + psi[l,j+1]) + w[l,j])

        for j in range(int((y_int_nodes+2)/2),int(3*(y_int_nodes+2)/4)): # left wall inlet
            psi_temp[0,j] = (4*psi_temp[1,j] - psi_temp[2,j] + 2*dx*v[0,j]) / 3
    
        for j in range(int((y_int_nodes+2)/4),int((y_int_nodes+2)/2)): # right wall outlet
            psi_temp[x_int_nodes+1,j] = (4*psi_temp[x_int_nodes,j] - psi_temp[x_int_nodes-1,j] - 2*dx*v[x_int_nodes+1,j]) / 3

        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                error = np.abs(psi_temp[l,j] - psi[l,j])
                if error >= max_error:
                    num_bad_nodes += 1
        iter += 1
        psi[:,:] = psi_temp[:,:]
    # print(iter)

    return psi

@jit
def solve(Lx, Ly, dx, dy, t_f, t_sol, dt, q, omega, nu, U, u_in):
    x_int_nodes = int(Lx / dx) - 1
    y_int_nodes = int(Ly / dy) - 1

    t_steps = int(t_f / dt)

    psi = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # streamfunction
    u = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # x velocity
    v = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # y velocity
    w = np.zeros((x_int_nodes+2, y_int_nodes+2, t_steps), dtype=np.double) # vorticity

    print(u.shape)
    u[:,y_int_nodes+1,:] = U # top surface moving plate BC
    u[0,int((y_int_nodes+2)/2):int(3*(y_int_nodes+2)/4),:] = u_in
    u[x_int_nodes+1,int((y_int_nodes+2)/4):int((y_int_nodes+2)/2),:] = u_in

    inlet_size = dy * (y_int_nodes+2)/2
    flow_rate = u_in * inlet_size 

    # initialize omega, first order centered differences
    for l in range(1, x_int_nodes+1):
        for j in range(1, y_int_nodes+1):
            w[l,j,0] = (v[l+1,j,0] - v[l-1,j,0]) / (2*dx) - (u[l,j+1,0] - u[l,j-1,0]) / (2*dy)

    psi[:, y_int_nodes+1, 0] = flow_rate # top wall
    psi[0, int(3*(y_int_nodes+2)/4):y_int_nodes+2, 0] = flow_rate # left wall above inlet
    psi[x_int_nodes+1, int((y_int_nodes+2)/2):y_int_nodes+2, 0] = flow_rate # right wall above outlet


    # psi[:,:,0] = GS(psi[:,:,0], w[:,:,0], u[:,:,0], v[:,:,0], x_int_nodes, y_int_nodes, dx, dy)
    psi[:,:,0] = SOR(psi[:,:,0], w[:,:,0], u[:,:,0], v[:,:,0], x_int_nodes, y_int_nodes, omega, dx, dy)

    # time iteration
    for n in range(t_steps-1):

        # omega boundary conditions

        # left and right boundaries
        for j in range(0,y_int_nodes+2):
            if j == 0:
                du_dy_L = (u[0,j+1,n] - u[0,j,n]) / dy
                du_dy_R = (u[x_int_nodes+1,j+1,n] - u[x_int_nodes+1,j,n]) / dy
            elif j == y_int_nodes+1:
                du_dy_L = (u[0,j,n] - u[0,j-1,n]) / dy
                du_dy_R = (u[x_int_nodes+1,j,n] - u[x_int_nodes+1,j-1,n]) / dy
            else:
                du_dy_L = (u[0,j+1,n] - u[0,j-1,n]) / (2*dy)
                du_dy_R = (u[x_int_nodes+1,j+1,n] - u[x_int_nodes+1,j-1,n]) / (2*dy)

            w[0,j,n] = -2 * (psi[1,j,n] - psi[0,j,n]) / dx**2 - 2 * v[0,j,n] / dx - du_dy_L # left wall
            w[x_int_nodes+1,j,n] = -2 * (psi[x_int_nodes,j,n] - psi[x_int_nodes+1,j,n]) / dx**2 + 2 * v[x_int_nodes+1,j,n] / dx - du_dy_R # right wall
        
        # top and bottom boundaries
        for l in range(0,x_int_nodes+2):
            w[l,0,n] = -2 * (psi[l,1,n] - psi[l,0,n]) / dy**2 + 2 * u[l,0,n] / dy # bottom wall
            w[l,y_int_nodes+1,n] = -2 * (psi[l,y_int_nodes,n] - psi[l,y_int_nodes+1,n]) / dy**2 - 2 * u[l,y_int_nodes+1,n] / dy # top wall
        
        # first solve for omega_n+1 without blending
        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                w[l,j,n+1] = (w[l,j,n] - (u[l,j,n]*dt/(2*dx)) * (w[l+1,j,n] - w[l-1,j,n]) - (v[l,j,n]*dt/(2*dy)) * (w[l,j+1,n] - w[l,j-1,n])
                              + (nu*dt/dx**2) * (w[l-1,j,n] - 2*w[l,j,n] + w[l+1,j,n]) + (nu*dt/dy**2) * (w[l,j-1,n] - 2*w[l,j,n] + w[l,j+1,n]))
        
        if q > 0: # blended discretization
            for l in range(2,x_int_nodes-1):
                for j in range(2,y_int_nodes-1):
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

        # psi[:,:,n+1] = GS(psi[:,:,n], w[:,:,n+1], u[:,:,n+1], v[:,:,n+1], x_int_nodes, y_int_nodes, dx, dy)
        psi[:,:,n+1] = SOR(psi[:,:,n], w[:,:,n+1], u[:,:,n+1], v[:,:,n+1], x_int_nodes, y_int_nodes, omega, dx, dy)

        #print(n+1)

        # solve for velocity field, centered differences
        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                u[l,j,n+1] = (psi[l,j+1,n+1] - psi[l,j-1,n+1]) / (2*dy)
                v[l,j,n+1] = -(psi[l+1,j,n+1] - psi[l-1,j,n+1]) / (2*dx)
          
    return (u, v, w, psi)