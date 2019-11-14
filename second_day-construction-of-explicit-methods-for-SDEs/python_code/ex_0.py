import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint


def f(y, t):
    s, i = y
    dsdt = -beta * s * i / N + (b + gamma) * i
    didt = beta * s * i / N - (b + gamma) * i
    return [dsdt, didt]


N = 100.0  # maximum population size
delta_t = 0.01

#beta = 1.0
#gamma = 0.25
#b = 0.25  # birth rate
beta = .2
gamma = 0.5
b = 0.5  # birth rate
num_of_realizations = 10
# odeint parameters
y_0 = np.array([98, 2])
t = np.linspace(0, 20, 2000)
sol = odeint(f, y_0, t)

final_time = 2000
infected = np.zeros(final_time)
infected[0] = 2
possible_transitions = np.array([1, -1, 0])
plt.figure(figsize=(6, 3))

for k in np.arange(5):
    for delta_t_i in range(final_time - 1):
        birth = False
        death = False
        if 0 < infected[delta_t_i] < N-1:
            # Is there a birth?
            birth_probability = beta * infected[delta_t_i] * (N - infected[delta_t_i]) / N * delta_t
            death_probability = (b + gamma) * infected[delta_t_i] * delta_t
            complement_probability = 1.0 - (birth_probability + death_probability) * delta_t
            if np.random.rand() <= birth_probability:
                birth = True
            if np.random.rand() <= death_probability:
                death = True
            transition = 1 * birth - 1 * death
            infected[delta_t_i + 1] = infected[delta_t_i] + transition
        # The evolution stops if we reach $0$ or $N$.
        else:
            infected[delta_t_i + 1] = infected[delta_t_i]
    plt.plot(t, infected)

plt.plot(t, sol[:, 1], ls='--', color='black', lw=2)
plt.title('R_zero = 0.2')
plt.xlabel('time')
plt.ylabel('Infected Individuals')
plt.show()
