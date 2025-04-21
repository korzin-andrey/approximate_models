import numpy as np
import scipy
import networkx as nx
from collections import Counter, defaultdict


class SeirMeanFieldModel():
    def __init__(self, G):
        '''
        :G networkx graph for contact network
        '''
        self.G = G
        
    def _initialize_node_status_(G, initial_infecteds, initial_recovereds = None):
        if initial_recovereds is None:
            initial_recovereds = []
        intersection = set(initial_infecteds).intersection(set(initial_recovereds))
        if  intersection:
            raise Exception("{} are in both initial_infecteds and initial_recovereds".format(intersection))

        status = defaultdict(lambda : 'S')
        for node in initial_infecteds:
            if not G.has_node(node):
                raise Exception("{} not in G".format(node))
            status[node] = 'I'
        for node in initial_recovereds:
            if not G.has_node(node):
                raise Exception("{} not in G".format(node))
            status[node] = 'R'
        return status

    def _get_Nk_Sk_Ek_Ik_Rk_as_arrays_(self, rho):
        Nk = Counter(dict(self.G.degree()).values())
        maxk = max(Nk.keys())
        Nk = np.array([Nk[k] for k in range(maxk+1)])

        if rho is None:
            rho = 1./self.G.order()
        Sk0 = (1-rho)*Nk
        Ik0 = rho*Nk
        Rk0 = 0*Nk
        Ek0 = 0*Nk
        return Nk, Sk0, Ek0, Ik0, Rk0

    @staticmethod
    def _dSEIR_heterogeneous_meanfield_(X, t, S0, Nk, tau, alpha, gamma):
        theta = X[0]
        dim_Nk = len(Nk)
        Ek = np.array(X[1:dim_Nk+1])
        Rk = np.array(X[dim_Nk+1:])
        ks = np.arange(len(Rk))
        Sk = S0*(theta**ks)
        Ik = Nk - Sk - Ek - Rk
        pi_I = ks.dot(Ik)/ks.dot(Nk)

        dRkdt = gamma*Ik
        dEkdt =Sk*ks*tau*pi_I*theta**ks - alpha*Ek
        dThetadt = - tau * pi_I * theta

        dX = np.concatenate(([dThetadt],dEkdt, dRkdt), axis=0)
        return dX


    def SEIR_heterogeneous_meanfield(self, Sk0, Ek0, Ik0, Rk0, tau, alpha, gamma, tmin = 0, tmax=100):
        if len(Sk0) != len(Ik0) or len(Sk0) != len(Rk0) or len(Sk0) != len(Ek0):
            raise Exception('length of Sk0, Ik0, and Rk0 must be the same')

        theta0=1
        Sk0 = np.array(Sk0)
        Ek0 = np.array(Ek0)
        Ik0 = np.array(Ik0)
        Rk0 = np.array(Rk0)
        Nk = Sk0 + Ek0 + Ik0 + Rk0
        X0 = np.concatenate(([theta0],Ek0, Rk0), axis=0)
        times = np.linspace(tmin, tmax, tmax+1)

        X = scipy.integrate.odeint(self._dSEIR_heterogeneous_meanfield_, X0, times,
                                args = (Sk0, Nk, tau, alpha, gamma))

        dim_Nk = len(Nk)
        theta = X.T[0]
        Ek = X.T[1:1+dim_Nk]
        Rk = X.T[1+dim_Nk:]
        # print(r"X: {}, Nk: {}, Rk: {}".format(X.shape, Nk.shape, Rk.shape))
        assert len(Ek) == len(Rk)
        ks = np.arange(len(Rk))
        L=(theta[None,:]**ks[:,None])
        Sk=Sk0[:,None]*L
        Ik = Nk[:,None]-Sk-Rk
        return times, sum(Sk), sum(Ek), sum(Ik), sum(Rk)
    
    def find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def seir_transform_event_times_to_days(self, t, S, E, I, R, tmax):
        indices = []
        for day in range(tmax):
            index = self.find_nearest_idx(t, day)
            indices.append(index)
        return t[indices], S[indices], E[indices], I[indices], R[indices]

    def simulate(self, beta=0.02, alpha=0.1, gamma=0.1, rho=0.005, tmin = 0, tmax=150):
        _, Sk0, Ek0, Ik0, Rk0 = self._get_Nk_Sk_Ek_Ik_Rk_as_arrays_(rho)
        t, S, E, I, R = self.SEIR_heterogeneous_meanfield(Sk0, Ek0, Ik0, Rk0, beta, alpha, gamma, tmin, tmax) 
        t, S, E, I, R = self.seir_transform_event_times_to_days(t, S, E, I, R, tmax)
        return t, S, E, I, R
    