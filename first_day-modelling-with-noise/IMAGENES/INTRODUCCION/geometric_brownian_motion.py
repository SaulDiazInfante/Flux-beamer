"""
    This class solve explicitly the geometric brownian SDE proposal in the 
    almost all books, see e.g. Roberts
        \begin{equation}
        dX_t = \muX_t+\sigma X_tdB_t
    \end{equation}
"""
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

class SDEGeometricBrownianMotion:
    """
        Set the parameters and terms of the Stochastic Ginsburg-Landau equation.
       
    """
    def __init__(self, k=12, p=2, r=2, T0=0.0, T=6.0):
        self.k = k
        self.p = p
        self.r = r
        self.T0 = T0
        # self.N=10.0**k
        # self.P=10.0**p
        # self.R=10.0**r
        self.N = 2.0 ** k
        self.P = 2.0 ** p
        self.R = 2.0 ** r
        self.T = T
        self.dt = self.T / np.float(self.N)
        self.IndexN = np.arange(self.N + 1)
        # set of index to Ito integral
        self.tau = self.IndexN[0:self.N + 1:self.P]
        self.t = np.linspace(0, self.T, self.N + 1)
        self.Dt = np.float(self.R) * self.dt
        self.L = self.N / self.R
        self.mu=1.0
        self.sigma=1.0
        self.x_zero=1.5

    def initialize_mesh(self, k, p, r, T0, T):
        """
            Set stencil parameters
        """
        self.k = k
        self.p = p
        self.r = r
        self.T0 = T0
        # self.N=10.0**k
        # self.P=10.0**pGinzburgLandau
        # self.R=10.0**r
        self.N = 2.0 ** k
        self.P = 2.0 ** p
        self.R = 2.0 ** r
        self.T = T
        self.dt = self.T / np.float(self.N)
        self.IndexN = np.arange(self.N + 1)
        # set of index to Ito integral
        self.tau = self.IndexN[0:self.N + 1:self.P]
        self.t = np.linspace(0, self.T, self.N + 1)
        self.Dt = np.float(self.R) * self.dt
        self.L = self.N / self.R
        # diffusion part
        self.DistNormal = np.random.randn(np.int(self.N))
        self.dW = np.sqrt(self.dt) * self.DistNormal
        self.W = np.cumsum(self.dW)
        self.W = np.concatenate(([0], self.W))

    def set_parameters_bm(self, mu, sigma, x_zero):
        """
        Set parameters of Ginsburg-Landau SDE.
        """
        self.mu = mu
        self.sigma = sigma
        self.x_zero = x_zero

    def a(self, x):
        """
            The drifft term of the SDE.
        """
        mu = self.mu

        return mu * x

    def b(self, x):
        """
            The diffusion term.
        """
        return self.sigma * x


class NumericsSDEGeometricBrowninanMotion(SDEGeometricBrownianMotion):
    """
        Numerics for the Stochastic Linzburg Landau equation.
    """

    def EM(self, flag=0):
        Dt = self.Dt
        L = self.L
        R = self.R
        if flag == 1:
            Dt = self.dt
            L = self.N
            R = 1
        self.Xem = np.zeros(L + 1)
        self.Xem[0] = self.x_zero
        for j in np.arange(L):
            self.Winc = np.sum(self.dW[R * (j):R * (j + 1)])
            increment = self.Xem[j] + Dt * self.a(self.Xem[j]) + self.Winc * self.b(self.Xem[j])
            self.Xem[j + 1] = increment
        xem = self.Xem
        return xem
