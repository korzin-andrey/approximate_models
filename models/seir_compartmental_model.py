'''
    Module contatins compartmental SEIR model simulation class.
'''

import numpy as np
import scipy


class SeirModel():
    def __init__(self, population: int = 10**6):
        '''
        population: population size
        '''
        self.population = population

    def __deriv(self, y, t, alpha, beta, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def simulate(self, alpha=0.1, beta=0.01, gamma=0.1, init_inf_frac=0.0005, init_rec_frac=0, tmax: int = 150):
        '''
        alpha: rate of progression from exposed to infectious
        beta: transmission rate
        gamma: recovery rate
        init_inf_frac: fraction of initially infected
        init_rec_frac: fraction of initially recovered
        '''
        E0 = 0
        I0 = init_inf_frac
        R0 = init_rec_frac
        S0 = 1 - I0 - R0
        y0 = S0, E0, I0, R0
        t = np.linspace(0, tmax, tmax)
        S, E, I, R = scipy.integrate.odeint(self.__deriv, y0, t,
                                            args=(alpha, beta, gamma)).T * self.population
        return t, S, E, I, R
    
    