import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as io

def wave(U, A, lam, Lx, dx, x1, x2, t):
    x_steps = int(Lx/dx)
    x_range = np.linspace(0, Lx, x_steps)
    eta = np.zeros(x_steps) # IC

    for i in range(x_steps):
        if x_range[i] < x1 + U*t or x_range[i] > x2 + U*t:
            eta[i] = 0
        else:
            eta[i] = A * np.sin( 2 * np.pi * (x_range[i] - U*t) / lam)

    return eta

def create_GIF(x, y, dt, H):
    filenames = []
    for i in range(len(y)):
        plt.plot(x, H+y[i], label='t = ' + str("{:0.2f}".format(dt*i)))
        plt.xlabel('Length (m)')
        plt.ylabel('H + ' + r'$\eta$' + ' (m)')
        plt.title('Lax-W with Periodic BCs, CFL = 1.0')
        plt.grid()
        plt.legend(loc=2)
        filename = f'{i}.png'
        filenames.append(filename)

        plt.savefig(filename)
        plt.close()

    # build gif
    with io.get_writer('j.gif', mode='I', duration=dt) as writer:
        for filename in filenames:
            image = io.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
