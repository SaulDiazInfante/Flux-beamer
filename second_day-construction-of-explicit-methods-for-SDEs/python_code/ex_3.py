import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint


def f(y, t):
    i = y
    #dsdt = -beta * s * i / N + (b + gamma) * i
    didt = beta * (N - i) * i / N - (b + gamma) * i
    return didt

def g(y):
    sigma_square = beta / N * y * (N - y) + (b + gamma) * y
    return np.sqrt(sigma_square)

N = 100.0           # maximum population size
delta_t = 0.001
#beta = 1.0
#gamma = 0.25
#b = 0.25  # birth rate
beta = 1.0
gamma = 0.5
b = 0.6     # birth rate
num_of_realizations = 5
# odeint parameters
y_0 = 2.0
array_size = np.int(25.0 / delta_t)
time = np.linspace(0, 25, array_size)
sol = odeint(f, y_0, time)
infected = np.zeros([array_size])
infected_0 = 2.0
plt.figure(figsize=(6, 3))
for k in np.arange(num_of_realizations):
    normal_sampler = np.zeros(array_size)
    normal_sampler[1:] = np.sqrt(delta_t) * np.random.randn(array_size -1)
    winner_inc = np.cumsum(normal_sampler)
    infected[0] = infected_0
    for i in np.arange(array_size - 1):
        delta_w_i = winner_inc[i+1] - winner_inc[i]  
        euler_i = infected[i] + f(infected[i], 0) * delta_t \
        + g(infected[i]) * delta_w_i
        infected[i + 1] = euler_i
    plt.plot(time, infected)
# odeint parameters
plt.plot(time, sol, ls='--', color='black', lw=2)
plt.title('R_zero = 0.90909090')
plt.xlabel('time')
plt.ylabel('Infected Individuals')
plt.show()