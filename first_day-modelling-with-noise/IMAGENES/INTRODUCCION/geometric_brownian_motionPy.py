import numpy as np
import matplotlib.pyplot as plt
from geometric_brownian_motion import NumericsSDEGeometricBrowninanMotion

mu = 1.0
sigma = .25
x_zero = 1.5

k = 12
p = 4
r = p
T0 = 0.0
T = 6.0

gbm = NumericsSDEGeometricBrowninanMotion()
gbm.initialize_mesh(k, p, r, T0, T)
gbm.set_parameters_bm(mu, sigma, x_zero)

h = gbm.Dt
strEM = 'EM h=' + str(h)
#
x_em = gbm.EM(flag=1)
#
plt.figure()
# plt.axis([0, 0.25*T,-5.0, 5.0])
plt.plot(gbm.t, x_em,
         color='blue',
         marker='.',
         alpha=0.8,
         # lw=1,
         ms=1,
         mfc='gray',
         label="noise"
         )
plt.legend(loc=0)
plt.grid()
plt.show()
data = np.array([gbm.t, x_em])
data = data.transpose()
np.savetxt('gbm.csv', data, delimiter=',')
