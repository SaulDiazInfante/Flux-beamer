import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint


def f(y, t):
    s, i = y
    dsdt = -beta * s * i / N + (b + gamma) * i
    didt = beta * s * i / N - (b + gamma) * i
    return [dsdt, didt]


N = 100.0   # maximum population size
delta_t = 0.01
#beta = 1.0
#gamma = 0.25
#b = 0.25  # birth rate
beta = 1.0
gamma = 0.5
b = 0.6     # birth rate
num_of_realizations = 5
# odeint parameters
y_0 = np.array([98, 2])
t = np.linspace(0, 20, 2000)
sol = odeint(f, y_0, t)

final_time = 25
infected_0 = 2
possible_transitions = np.array([1, -1, 0])
plt.figure(figsize=(6, 3))

for k in np.arange(num_of_realizations):
    t = []
    s = []
    i = []
    #
    t.append(0)
    s.append(N - infected_0)
    i.append(infected_0)
    j = 0
    while (i[j] > 0 and t[j] < final_time):
        u_1 = np.random.rand()
        u_2 = np.random.rand()
        den = (beta / N) * i[j] * s[j] + (b + gamma) * i[j]
        infection_prob = (beta * s[j] / N) / (beta * s[j] / N + b + gamma)
        # exponential random time
        t.append(t[j] - np.log(u_1) / den)
        if u_1 <= infection_prob:
            i.append(i[j] + 1)
            s.append(s[j] - 1)
        else:
            i.append(i[j] - 1)
            s.append(s[j] + 1)            
        j = j+1
    plt.plot(np.array(t), np.array(i))

# odeint parameters
y_0 = np.array([98, 2])
sol = odeint(f, y_0, t)
plt.plot(t, sol[:, 1], ls='--', color='black', lw=2)
plt.title('R_zero = 0.909090909090')
plt.xlabel('time')
plt.ylabel('Infected Individuals')
plt.show()