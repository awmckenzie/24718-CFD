import numpy as np
from numba import jit

@jit
def scheme(dx, dy, Lx, Ly, u, v, mu, t, dt, phi_0y, dphi_x0, dphi_xLy, dphi_Lxy, phi_0):
    
    CFL_x = u * dt / dx
    CFL_y = v * dt / dy
    eta_x = mu * dt / dx**2
    eta_y = mu * dt / dy**2

    if eta_x == 0.5*CFL_x**2:
        raise ValueError('eta_x = 0.5*CFL_x^2')
    if eta_y == 0.5*CFL_y**2:
        raise ValueError('eta_y = 0.5*CFL_y^2')

    print(CFL_x, CFL_y, eta_x, eta_y)
    print(dx**2/(2*mu), dt)
    print(dy**2/(2*mu), dt)

    x_int_nodes = int(Lx / dx) - 1
    y_int_nodes = int(Ly / dy) - 1
    t_steps = int(t/dt)

    print(t_steps, x_int_nodes+2, y_int_nodes+2)
    sol = np.zeros((t_steps, x_int_nodes+2, y_int_nodes+2), dtype=np.double)
    sol.fill(phi_0)

    # left edge Dirichlet BCs
    for j in range(0, y_int_nodes+2):
        if j * dy < Ly / 3:
            sol[:,0,j] = phi_0y[0]
        elif Ly / 3 <= j * dy <= 2 * Ly / 3:
            sol[:,0,j] = phi_0y[1]
        else:
            sol[:,0,j] = phi_0y[2]

    # solve for interior nodes
    for n in range(t_steps-1):
        for l in range(1, x_int_nodes+1):
            for j in range(1, y_int_nodes+1):
                sol[n+1,l,j] = (sol[n,l,j] - (u * dt / dx)* (sol[n,l,j] - sol[n,l-1,j]) - (v * dt / dy) * (sol[n,l,j] - sol[n,l,j-1])
                                + (mu*dt/dx**2) * (sol[n,l-1,j] - 2 * sol[n,l,j] + sol[n,l+1,j])
                                + (mu*dt/dy**2) * (sol[n,l,j-1] - 2 * sol[n,l,j] + sol[n,l,j+1]))

        # right edge Neumann BC, x=Lx
        sol[n+1,x_int_nodes+1,:] = sol[n+1,x_int_nodes,:] + dphi_Lxy * dx

        # bottom edge Neumann BCs, y=0
        sol[n+1,:,0] = sol[n+1,:,1] - dphi_x0 * dy

        # top edge Neumann BC, y=Ly
        sol[n+1,:,y_int_nodes+1] = sol[n+1,:,y_int_nodes] + dphi_xLy * dy      

    return sol, dt