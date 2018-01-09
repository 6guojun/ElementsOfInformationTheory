#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:44:38 2018

@author: hollymandel
"""

import numpy as np
import random as rd

def makeTransMat(sz):
    # ord is the order of action - reciprocal to period
    trans = np.random.rand(sz,sz)
    trans = np.asmatrix(trans)
    trans = np.asmatrix(trans)
    trans_rowSums = trans*np.matrix.transpose(np.matrix(np.ones(sz)))
    trans_rowSums = np.asarray(np.matrix.transpose(trans_rowSums))
    trans_rowSumsInv = np.power(trans_rowSums,-1)
    trans_rowSumsInv = np.asmatrix(np.diag(trans_rowSumsInv[0]))
    trans = trans_rowSumsInv*trans
    return trans

samples = 10
size1 = 5
size2 = 3
mapDown = np.random.randint(size2,size=(1,size1))
states1 = np.random.rand(samples,size1)
states2 = np.random.rand(samples,size2)
trans1 = makeTransMat(size1)

