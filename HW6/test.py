import numpy as np

t_f = 5
dt = 0.1
t_steps = int(t_f/dt)
arr = np.zeros(t_steps)

for n in range(t_steps):
    print(arr[n])