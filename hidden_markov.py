#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:44:38 2018

@author: hollymandel
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt

def makeTransMat(sz):
    # weights matrix with 0s on the diagonal (to match 4.3 in T-C)
    # symmetric. evaluate as input^t * P
    # it turns out that generating such a matrix that is also doubly stochastic
    # is nontrivial! So we are not normalizing here
    trans = np.random.uniform(0,1,size=(sz,sz))
    trans = trans + trans.T
    trans = trans - np.diag(np.diag(trans))
    
#    trans = np.zeros((sz,sz))
#    for i in range(0,sz):
#        leftOver = 1-np.sum(trans[i,0:i])
#        rowVals = np.random.rand(1,sz-(i+1)) # change distribution! no -!
#        rowVals = leftOver*rowVals/np.sum(rowVals)
#        trans[i] = np.concatenate((trans[i,0:i+1],rowVals[0]))
#        trans[:,i] = trans[i].T
#             
             
#    lastRowSum = np.sum(trans[:,sz-1])
#    lastRowHold = trans[sz-1]/lastRowSum
#    trans[sz-1] = lastRowHold
#    trans[:,sz-1] = lastRowHold
        
#    trans = np.random.rand(sz,sz)
#    transDiag = np.diag(np.diag(trans))
#    trans = trans - transDiag
#    trans = np.asmatrix(trans)
#    trans_rowSums = trans*np.matrix.transpose(np.matrix(np.ones(sz)))
#    trans_rowSums = np.asarray(np.matrix.transpose(trans_rowSums))
#    trans_rowSumsInv = np.power(trans_rowSums,-1)
#    trans_rowSumsInv = np.asmatrix(np.diag(trans_rowSumsInv[0]))
#    trans = trans_rowSumsInv*trans
    return trans

def getStationaryDistribution(A):
    # given an n x n transition matrix with 0s along the diagonal, returns the 
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

transCumulative = np.zeros(trans1.shape)
for i in range(0,size1):
    for j in range(0,size1):
        for k in range(0,j+1):
            transCumulative[i,j] += trans1[i,k]
            
mapDown = np.random.randint(size2,size=(size1))


# prediction
#p1 = getStationaryDistribution(trans1)
#p2 = np.zeros(size2)
#for i in range(0,size2):
#    for j in range(0,size1):
#        if mapDown[j] == i:
#            p2[i] += p1[j]
#        
#print(p1)
#print(mapDown)
#print(p2)

# simulation

        
stateChoices = np.random.rand(samples-1,population)
states = np.random.randint(0,size1, size=(1,population))
states = np.vstack((states[0],np.zeros((samples-1,population))))
for i in range(1,samples):
    for j in range(0,population):
        getChoice = stateChoices[i-1,j]
        getProb = transCumulative[states[i-1,j]]
        probCheck = (getProb > getChoice)
        probCheck = np.nonzero(probCheck)
        states[i,j] = probCheck[0][0]
        
states = states.astype(int) 
      
fracStates = np.zeros((samples,size1))
for i in range(0,samples):
    fracStates[i] = np.bincount(states[i])
        
#fracStates = fracStates.cumsum(0)
#fracNorms = population*(range(0,samples)+np.ones((1,samples)))
fracStates = fracStates / population


plt.plot(fracStates)
plt.ylabel('some numbers')
plt.show()
plt.ylim((0,1))


