#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:44:38 2018

@author: hollymandel
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt

def makeWeightMat(sz):
    # weights matrix with 0s on the diagonal (to match 4.3 in T-C)
    # symmetric. evaluate as input^t * P
    # it turns out that generating such a matrix that is also doubly stochastic
    # is nontrivial! So we are not normalizing here
    weight = np.random.uniform(0,1,size=(sz,sz))
    weight = weight + weight.T
    weight = weight - np.diag(np.diag(weight))

    return weight

def getStationaryDistribution(A):
    # given an n x n weightition matrix with 0s along the diagonal, returns the 
    # stationary distribution, computer as in T-C 4.3
    n = A.shape[1]
    W = sum(A)*np.transpose(np.matrix(np.ones(n)))
    Wvect = A*np.transpose(np.matrix(np.ones(n)))
    stationaryDist = Wvect*(W**-1)
    
    return stationaryDist
   
population = 10000
samples = 15
size1 = 5
size2 = 3
weight1 = makeWeightMat(size1)
statDist = getStationaryDistribution(weight1)

weightCumulative = np.zeros(weight1.shape) # THIS IS WRONG
for i in range(0,size1):
    for j in range(0,size1):
        for k in range(0,j+1):
            weightCumulative[i,j] += weight1[i,k]
            
mapDown = np.random.randint(size2,size=(size1))
        
stateChoices = np.random.rand(samples-1,population)
states = np.random.randint(0,size1, size=(1,population))
states = np.vstack((states[0],np.zeros((samples-1,population))))
for i in range(1,samples):
    for j in range(0,population):
        getChoice = stateChoices[i-1,j]
        getProb = weightCumulative[states[i-1,j]]
        probCheck = (getProb > getChoice)
        probCheck = np.nonzero(probCheck)
        states[i,j] = probCheck[0][0]
        
states = states.astype(int) 
      
fracStates = np.zeros((samples,size1))
for i in range(0,samples):
    getBinCount = np.bincount(states[i])
    gbSz = getBinCount.size
    if gbSz < size1:
        zeroPad = np.zeros((1,size1-gbSz))
        getBinCount = np.concatenate((getBinCount,zeroPad[0]))                            
    fracStates[i] = getBinCount
    
fracStates = fracStates / population


plt.plot(fracStates)
plt.ylabel('some numbers')
plt.show()
plt.ylim((0,1))


