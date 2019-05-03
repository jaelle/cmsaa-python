import numpy as np
from scipy.optimize import curve_fit
from math import exp

import matplotlib.pyplot as plt


def attentional_bias(expRTs):
    
    return (2000 - np.array(expRTs))/2000


def optimize_prioritymap(attended_location, x, y, init_vals, min_bounds, max_bounds):

    pm = PriorityMap(attended_location)

    (best_vals,covar) = curve_fit(pm.standard, x, y, p0=init_vals, bounds=(min_bounds,max_bounds))

    return best_vals

def rmse(xs,pm,experimental):
    error = 0
    
    i = 0
    for x in xs:
        error += (experimental[i] - pm[x]) ** 2
        i += 1
        
    return error

def plot_results(x,y,test_y,pm, filename):
    # range is from the first x value to the last one
    degrees = np.arange(x[0],x[len(x)-1],1)


    plt.plot(x,y,'yo')
    plt.plot(x,test_y,'kx')
    plt.plot(degrees,pm.goalmap,'b')
    plt.plot(degrees,pm.saliencymap,'r')
    plt.plot(degrees,pm.prioritymap,'g')

    if filename != '':
        plt.savefig(filename,format='png')
    else:
        plt.show()

    plt.close()

class GoalMap:
    def __init__(self, attended_location):
        self.attended_location = attended_location
    
    def standard(self, x, mag, stdev):
        # gaussian equation
        return mag * np.exp(-abs(self.attended_location-x)**2 / (2*stdev**2))
    
    # shifts the goal map up or down by the minshift value
    def standard_minshift(self, x, mag, stdev, minshift):
        return minshift + standard(x,stdev, mag)

class SaliencyMap:
    def __init__(self, attended_location):
        self.attended_location = attended_location
    
    def standard(self, x, mag, stdev):
        # inverted gaussian equation
        return mag - mag * np.exp(-abs(self.attended_location-x)**2 / (2*stdev**2))
    
    # returns a constant attentional bias at every location in the map range
    def constant(self, x, value):
        return [value]*len(x)
    
    # returns a map where each degree location in the map range represents the probability of a sound at that location.
    # Probabilities are learned from the data
    def ibl(self, x, attended_location, data):
        
        # TODO: replace this with IBL algorithm
        return [0]*len(x)

class PriorityMap:
    def __init__(self, attended_location):
        self.attended_location = attended_location
        
    def standard(self, x, gm_mag, gm_stdev, sm_mag, sm_stdev):
        
        gm = GoalMap(self.attended_location)
        sm = SaliencyMap(self.attended_location)
        
        self.goalmap = gm.standard(x, gm_mag, gm_stdev)
        self.saliencymap = sm.standard(x, sm_mag, sm_stdev)
        self.prioritymap = self.goalmap + self.saliencymap
        
        return self.prioritymap

