import numpy as np
import funcs as fn

def CN(dx, dy, dt, Lx, Ly, t, T_0y, T_Lxy, T_x0, T_xLy, T_0, a, TDMA=False):

    b = (a*dt)/(2*dx**2) # beta constant
    p = (a*dt)/(2*dy**2) # phi constant

    x_int_nodes = int(Lx/dx) - 1
    y_int_nodes = int(Ly/dy) - 1

    Q = fn.block_tridiag(-p, -b, 1+2*b+2*p, x_int_nodes, y_int_nodes) # coeff matrix applied to 
    T_1 = np.zeros([x_int_nodes+2, y_int_nodes+2]) # x=rows, y=cols, transpose later
    T_1.fill(T_0) # init condition

    T_1 = np.zeros([x_int_nodes+2, y_int_nodes+2])
    T_1[:, y_int_nodes+1] = T_xLy
    T_1[:, 0] = T_x0
    T_1[0, :] = T_0y
    T_1[x_int_nodes+1, :] = T_Lxy

    T_2 = np.zeros([x_int_nodes+2, y_int_nodes+2])
    T_2[:, y_int_nodes+1] = T_xLy
    T_2[:, 0] = T_x0
    T_2[0, :] = T_0y
    T_2[x_int_nodes+1, :] = T_Lxy
    
    for i in range(int(t/dt)):
        T_1[:, y_int_nodes+1] = T_xLy
        T_1[:, 0] = T_x0
        T_1[0, :] = T_0y
        T_1[x_int_nodes+1, :] = T_Lxy
        B = np.zeros([x_int_nodes+2, y_int_nodes+2]) # full solution vector
        Bn = np.zeros([x_int_nodes, y_int_nodes])

        Bn[:, 0] += p*T_x0
        Bn[:, y_int_nodes-1] += p*T_xLy
        Bn[0, :] += b*T_0y
        Bn[x_int_nodes-1, :] += b*T_Lxy

        B[0, :] = T_0y
        B[x_int_nodes+1, :] = T_Lxy
        B[:, y_int_nodes+1] = T_xLy
        B[:, 0] = T_x0
        B[1:x_int_nodes+1, 1:y_int_nodes+1] = (np.multiply(1 - 2*b - 2*p, T_1[1:x_int_nodes+1, 1:y_int_nodes+1]) # build F_n
                                              + np.multiply(b, T_1[2:x_int_nodes+2, 1:y_int_nodes+1] + T_1[0:x_int_nodes, 1:y_int_nodes+1])
                                              + np.multiply(p, T_1[1:x_int_nodes+1, 2:y_int_nodes+2] + T_1[1:x_int_nodes+1, 0:y_int_nodes]))

        B_int = B[1:x_int_nodes+1, 1:y_int_nodes+1]
        Q_inv = np.linalg.inv(Q)
        T_2[1:x_int_nodes+1, 1:y_int_nodes+1] = np.dot(Q_inv, B_int.flatten('F') + Bn.flatten('F')).reshape((x_int_nodes, y_int_nodes), order='F')

        T_1 = T_2
    return T_1.T

