import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time as t 

# constants
H = 1.0 * 10**(-3) # m
rho = 1000 # kg/m^3
mu = 0.0016 # kg/m*s
time_pts = np.array([0, 0.0005, 0.1, 0.25, 0.5, 1.0])
time = 1.0 # s

# spatial and temporal step sizes
y_step = 0.02 * 10**(-3) # m
t_step = 0.0001 # s

# inital condition
u_init = 0 # m/s, at time 0

#boundary conditions
u_0 = 0 # m/s, at the stationary wall
u_H = 20 * 10**(-3) # m/s, at the moving wall

def TDMAsolver(a, b, c, d): # https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

def set_tridiag(n, beta):
    m = np.zeros([n, n]) # init coeff matrix
    diag1 = np.empty(n - 1)
    diag2 = np.empty(n)
    diag3 = np.empty(n - 1)
    diag1.fill(-beta)
    diag2.fill(1+2*beta)
    diag3.fill(-beta)
    m = m + np.diag(diag1, k=-1) + np.diag(diag2) + np.diag(diag3, k=1)
    return m

def FTCS_explicit(t, y, dt, dy, mu, rho, u_0, u_H, u_init):
    b = (mu*dt)/(rho*(dy**2)) # beta const.

    num_y_steps = int(y / dy)
    num_t_steps = int(t / dt)

    u = np.empty([num_t_steps, num_y_steps]) # solution matrix
    u[0].fill(u_init) # set initial condition

    for n in range(num_t_steps):
        if not(n==0):
            u[n, 0] = u_0 # set boundary condition at y=0
            u[n, num_y_steps-1] = u_H # set boundary condition at y=H

    for n in range(num_t_steps-1):
        for j in range(1, num_y_steps-1): # interior nodes
            u[n+1, j] = u[n, j] + b*(u[n, j+1] - 2*u[n, j] + u[n, j-1])
    
    return(u)

def FTCS_implicit(t, y, dt, dy, mu, rho, u_0, u_H, u_init):
    b = (mu*dt)/(rho*(dy**2)) # beta constant

    num_y_steps = int(y / dy)
    num_t_steps = int(t / dt)

    A = np.zeros([num_y_steps, num_y_steps]) # init coeff Aatrix
    diag1 = np.empty(num_y_steps - 1)
    diag2 = np.empty(num_y_steps)
    diag3 = np.empty(num_y_steps - 1)
    diag1.fill(-b)
    diag2.fill(1+2*b)
    diag3.fill(-b)
    A = A + np.diag(diag1, k=-1) + np.diag(diag2) + np.diag(diag3, k=1)

    A_inv = np.linalg.inv(A) # invert matrix, {u} = [A]^-1 * {b}

    u = np.zeros([num_t_steps, num_y_steps]) # solution matrix
    u[0, :] = u_init # set inital condition vector

    bc = np.zeros(num_y_steps) # boundary vector
    bc[0] = u_0 # set boundary condition at y=0
    bc[num_y_steps-1] = u_H # set boundary condition at y=H

    for i in range(num_t_steps-1):
        B = u[i, :] + b*bc # sum known information
        u[i+1, :] = np.dot(A_inv, B) # solve for u
    return(u)

def C_N(t, y, dt, dy, mu, rho, u_0, u_H, u_init, TDMA):
    b = (mu*dt)/(2*rho*(dy**2)) # beta constant

    num_y_steps = int(y / dy)
    num_t_steps = int(t / dt)

    A = np.zeros([num_y_steps, num_y_steps]) # init coeff Aatrix
    diag1 = np.empty(num_y_steps - 1)
    diag2 = np.empty(num_y_steps)
    diag3 = np.empty(num_y_steps - 1)
    diag1.fill(-b)
    diag2.fill(1+2*b)
    diag3.fill(-b)
    A = A + np.diag(diag1, k=-1) + np.diag(diag2) + np.diag(diag3, k=1)
    if not TDMA:
        A_inv = np.linalg.inv(A) # invert matrix, {u} = [A]^-1 * {b}
    rhs_diag = np.zeros([num_y_steps, num_y_steps])
    diag4 = np.empty(num_y_steps - 1)
    diag5 = np.empty(num_y_steps)
    diag6 = np.empty(num_y_steps - 1)
    diag4.fill(b)
    diag5.fill(1-2*b)
    diag6.fill(b)
    rhs_diag = rhs_diag + np.diag(diag4, k=-1) + np.diag(diag5) + np.diag(diag6, k=1)

    u = np.empty([num_t_steps, num_y_steps])
    u[0, :] = u_init # set inital condition vector

    bc = np.zeros(num_y_steps) # boundary vector
    bc[0] = u_0 # set boundary condition at y=0
    bc[num_y_steps - 1] = u_H # set boundary condition at y=H

    for i in range(num_t_steps - 1):
        B = np.dot(rhs_diag, u[i, :]) + 2*b*bc # sum known information
        if not TDMA:
            u[i+1, :] = np.dot(A_inv, B) # solve for u
        else:
            u[i+1, :] = TDMAsolver(diag1, diag2, diag3, B)
    return u

def plot(results, y, method):
    fig, ax = plt.subplots()
    ax.plot(y, results[0], label='0.0s')
    ax.plot(y, results[1], label='0.05s')
    ax.plot(y, results[2], label='0.1s')
    ax.plot(y, results[3], label='0.25s')
    ax.plot(y, results[4], label='0.5s')
    ax.plot(y, results[5], label='1.0s')
    ax.legend()
    ax.set_title(method)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("u (m/s)")

    return fig, ax

y = np.linspace(0, H, num=int(H/y_step))

time1 = t.time()
u_FTCS_explicit = FTCS_explicit(time, H, t_step, y_step, mu, rho, u_0, u_H, u_init)
time2 = t.time()
u_FTCS_implicit = FTCS_implicit(time, H, t_step, y_step, mu, rho, u_0, u_H, u_init)
time3 = t.time()
u_CN = C_N(time, H, t_step, y_step, mu, rho, u_0, u_H, u_init, False)
time4 = t.time()
u_CN_fast = C_N(time, H, t_step, y_step, mu, rho, u_0, u_H, u_init, True)
time5 = t.time()

FTCS_explicit_time = time2 - time1
FTCS_implicit_time = time3 - time2
CN_time = time4 - time3
CN_fast_time = time5 - time4

print(FTCS_explicit_time, FTCS_implicit_time, CN_time, CN_fast_time)
FTCS_explicit_results = [u_FTCS_explicit[0],
                        u_FTCS_explicit[int(0.05/t_step) - 1],
                        u_FTCS_explicit[int(0.1/t_step) - 1],
                        u_FTCS_explicit[int(0.25/t_step) - 1],
                        u_FTCS_explicit[int(0.5/t_step) - 1],
                        u_FTCS_explicit[int(1.0/t_step) - 1]]

fig1, ax1 = plot(FTCS_explicit_results, y, "FTCS Explicit")

FTCS_implicit_results = [u_FTCS_implicit[0],
                        u_FTCS_implicit[int(0.05/t_step) - 1],
                        u_FTCS_implicit[int(0.1/t_step) - 1],
                        u_FTCS_implicit[int(0.25/t_step) - 1],
                        u_FTCS_implicit[int(0.5/t_step) - 1],
                        u_FTCS_implicit[int(1.0/t_step) - 1]]

fig2, ax2 = plot(FTCS_implicit_results, y, "FTCS Implicit")

CN_results = [u_CN[0],
             u_CN[int(0.05/t_step) - 1],
             u_CN[int(0.1/t_step) - 1],
             u_CN[int(0.25/t_step) - 1],
             u_CN[int(0.5/t_step) - 1],
             u_CN[int(1.0/t_step) - 1]]

fig3, ax3 = plot(CN_results, y, "Crank-Nicolson")

fig4, ax4 = plt.subplots()
ax4.plot(y, FTCS_explicit_results[5], label='FTCS explicit')
ax4.plot(y, FTCS_implicit_results[5], label='FTCS implicit')
ax4.plot(y, CN_results[5], label ='Crank-Nicolson')
ax4.legend()
ax4.set_title("Comparison of Methods at t = 1.0s")
ax4.set_xlabel("y (m)")
ax4.set_ylabel("u (m/s)")


fig5, ax5 = plt.subplots()
methods = ['FTCS Explicit', 'FTCS Implicit', 'C-N', 'C-N with TDMA']
times = [FTCS_explicit_time, FTCS_implicit_time, CN_time, CN_fast_time]
ax5.bar(methods, times)
ax5.set_title("Comparison of Methods CPU time")
ax5.set_xlabel("time (s)")
ax5.set_ylabel("method")
plt.show()