import numpy as np
import scipy
import networkx as nx
from collections import Counter, defaultdict


class SeirEbcmModel():
    def __init__(self, G):
        '''
        :G networkx graph for contact network
        '''
        self.G = G
        
    def get_Pk(self):
        '''
        Used in several places so that we can input a graph and then we
        can call the methods that depend on the degree distribution.

        :G networkx Graph

        :Returns dict ``Pk[k]`` is the proportion of nodes with degree ``k``.
        '''
        Nk = Counter(dict(self.G.degree()).values())
        Pk = {x: Nk[x]/float(self.G.order()) for x in Nk.keys()}
        return Pk


    def _dSEIR_EBCM_(self, X, t, N, beta, alpha, gamma, PGF, PGF_prime, PGF_prime_prime):
        # initial conditions
        theta = X[0]
        phi = X[1]
        psi = X[2]
        E = X[3]
        I = X[4]

        dtheta = -beta*psi
        dphi = -alpha*phi + (PGF_prime_prime(theta)*beta/PGF_prime(1))*psi
        dpsi = alpha*phi - (beta + gamma)*psi
        S = PGF(theta)
        dE = beta*psi*PGF_prime(theta) - alpha*E
        dI = alpha*E - gamma*I
        R = 1 - S - E - I
        return np.array([dtheta, dphi, dpsi, dE, dI])

    
    def SEIR_EBCM(self, N, rho, beta, alpha, gamma, PGF, PGF_prime, PGF_prime_prime, tmin, tmax):
        times = np.linspace(tmin, tmax, tmax+1)
        # theta_0 = 1, E_0 << 1, I_0 = rho << 1, ..., ...
        X0 = np.array([1 - rho, 1e-2, 1e-2, 0, rho])
        X = scipy.integrate.odeint(self._dSEIR_EBCM_, X0, times,
                                   args=(N, beta, alpha, gamma, PGF, PGF_prime, PGF_prime_prime))
        theta = X.T[0]
        S = N*PGF(theta)
        E = N*X.T[3]
        I = N*X.T[4]
        R = N-S-E-I
        assert np.all(I) > -1
        return times, S, E, I, R

    def simulate(self, beta=0.02, alpha=0.1, gamma=0.1, rho=0.005, tmin=0, tmax=150):
        '''
        Given network G and rho, calculates N, psihat, psihatPrime, and calls EBCM.
        '''
        Pk = self.get_Pk()
        N = self.G.order()

        def PGF(x):
            return sum(Pk[k]*x**k for k in Pk)

        def PGF_prime(x):
            return sum(k*Pk[k]*x**(k-1) for k in Pk)

        def PGF_prime_prime(x):
            return sum(k*(k-1)*Pk[k]*x**(k-2) for k in Pk)

        return self.SEIR_EBCM(N, rho, beta, alpha, gamma, PGF, PGF_prime, PGF_prime_prime, tmin, tmax)
