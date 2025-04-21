'''
Module contains SEIR network stochastic simulation using Gillespie algorithm 
from EoN(Epidemics on Networks) library.
'''

import EoN
import networkx as nx
import numpy as np
from collections import defaultdict


class SeirNetworkModel():
    def __init__(self, G):
        '''
        Parameters:
        :G networkx graph
        '''
        self.G = G
        
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

    def simulate(self, alpha=0.1, beta=0.02, gamma=0.1, rho=0.005, tmax=150):
        H = nx.DiGraph()  # Spontaneous transition
        H.add_edge('E', 'I', rate=alpha)   # Latency rate
        H.add_edge('I', 'R', rate=gamma)    # Recovery rate

        J = nx.DiGraph()  # Contact-based transmission
        J.add_edge(('I', 'S'), ('I', 'E'), rate=beta)  # Transmission rate per contact

        N = len(self.G.nodes)
        initial_status = defaultdict(lambda: 'S')
        for node in range(int(rho*N)):
            initial_status[node] = 'I'
            
        t, S, E, I, R = self.seir_transform_event_times_to_days(*EoN.Gillespie_simple_contagion(self.G, H, 
                                       J, initial_status,
                                       return_statuses=('S', 'E', 'I', 'R'), tmax=tmax), tmax)
        return t, S, E, I, R
                                
