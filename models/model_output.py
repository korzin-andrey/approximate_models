"""Module for storing epidemic models output data

This module contains several classes for storing epidemic models
output data. During initialization you should pass to __init__() 
arrays of corresponding compartments. Classes estimates daily and 
weekly incidence values based on model type. 

Typical usage example:
    # assume you have arrays of time (t), susceptible(S), 
    # exposed(E), infected(I) and recovered(R)
    SEIR_output = SEIRModelOutput(t, S, E, I, R)
    weekly_incidence = SEIR_output.weekly_incidence
"""

import numpy as np


class SEIRModelOutput:
    '''
      Stores output data for SEIR compartmental model.

    '''

    def __init__(self, t, S, E, I, R):
        self.t = t
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.daily_incidence = None
        self.weekly_incidence = None
        self.calculate_incidence()

    def pad_array_to_multiple_of_seven(self, arr):
        '''
        Auxiliary function used for padding array of daily data by zeroes for converting
        to weekly data
        '''
        current_size = len(arr)
        new_size = (current_size + 6) // 7 * 7
        padding_needed = new_size - current_size
        padded_array = np.pad(arr, (0, padding_needed),
                              mode='constant', constant_values=0)
        return padded_array

    def calculate_incidence(self):
        self.incidence = [0 if index == 0 else ((self.E[index-1] - self.E[index]) -
                                                (self.S[index] - self.S[index-1])) for index in range(len(self.S))]
        daily_incidence_padded = self.pad_array_to_multiple_of_seven(
            self.incidence)
        self.weekly_incidence = daily_incidence_padded.reshape(
            -1, 7).sum(axis=1)
