import numpy as np
import funcs as fn

def ADI(dx, dy, dt, Lx, Ly, t, dT_0y, dT_Lxy, dT_x0, T_xLy, T_0, a, TDMA=False):

    b = (a*dt)/(2*dx**2) # beta constant, 1/2 dt compared to C-N
    p = (a*dt)/(2*dy**2) # phi constant, 1/2 dt compared to C-N

    x_int_nodes = int(Lx/dx) - 1 # number of interior nodes in x
    y_int_nodes = int(Ly/dy) - 1 # number of interior nodes in y

    Qx = fn.tridiag(-b, 1+2*b, x_int_nodes) # coeff matrix for x, 1D implicit
    Qy = fn.tridiag(-p, 1+2*p, y_int_nodes) # coeff matrix for y, 1D implicit

    T_1 = np.zeros([x_int_nodes+2, y_int_nodes+2]) # x=rows, y=cols, transpose later
    T_1.fill(T_0)
    T_2 = np.zeros([x_int_nodes+2, y_int_nodes+2])
    result = [None]*int(t/dt)
    for i in range(int(t/dt)):
        # n -> n + 1/2
        B = np.zeros([x_int_nodes+2]) # full x vector
        Bx = np.zeros([x_int_nodes]) # x BC vector at t = n + 1/2
        for j in range(1, y_int_nodes+1):
            B[1:x_int_nodes+1] = T_1[1:x_int_nodes+1, j] + np.multiply(p, T_1[1:x_int_nodes+1, j+1] - np.multiply(2, T_1[1:x_int_nodes+1, j]) + T_1[1:x_int_nodes+1, j-1]) # explicit along y
            B[0] = T_1[1, j] - dT_0y*dx # x = 0
            B[x_int_nodes+1] = T_1[x_int_nodes, j] + dT_Lxy*dx
            Bx[0] = b*(T_1[1, j] - dT_0y*dx)
            Bx[x_int_nodes-1] = b*(T_1[x_int_nodes, j] + dT_Lxy*dx)

            if not TDMA:
                Qx_inv = np.linalg.inv(Qx)
                T_2[1:x_int_nodes+1, j] = np.dot(Qx_inv, B[1:x_int_nodes+1] + Bx)
            else:
                T_2[1:x_int_nodes+1, j] = fn.TDMAsolver(np.diag(Qx, -1), np.diag(Qx, 0), np.diag(Qx, 1), B[1:x_int_nodes+1] + Bx)

        T_2[:, 0] = T_2[:, 1] - dT_x0*dy
        T_2[:, y_int_nodes+1] = T_xLy
        T_1 = T_2
    
        # n + 1/2 -> n + 1
        B = np.zeros([y_int_nodes+2]) # full y vector
        By = np.zeros([y_int_nodes]) # y BC vector at t = n + 1
        for l in range(1, x_int_nodes+1):
            B[1:y_int_nodes+1] = T_1[l, 1:y_int_nodes+1] + np.multiply(b, T_1[l+1, 1:y_int_nodes+1] - np.multiply(2, T_1[l, 1:y_int_nodes+1]) + T_1[l-1, 1:y_int_nodes+1]) # explicit along x
            B[0] = T_1[l, 1] - dT_x0*dy
            B[y_int_nodes+1] = T_xLy
            By[0] = p*(T_1[l, 1] - dT_x0*dy)
            By[y_int_nodes-1] = p*T_xLy

            if not TDMA:
                Qy_inv = np.linalg.inv(Qy)
                T_2[l, 1:y_int_nodes+1] = np.dot(Qy_inv, B[1:y_int_nodes+1] + By)
            else:
                T_2[l, 1:y_int_nodes+1] = fn.TDMAsolver(np.diag(Qy, -1), np.diag(Qy, 0), np.diag(Qy, 1), B[1:y_int_nodes+1] + By)
                
        T_2[0, :] = T_2[1, :] - dT_0y*dx
        T_2[x_int_nodes+1, :] = T_2[x_int_nodes, :] + dT_Lxy*dx
        T_1 = T_2
        result[i] = T_1.T
    return result