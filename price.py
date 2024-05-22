from matplotlib.pyplot import xlabel, ylabel
import numpy as np
from analysis import *
from newGenerator import *
# from partitionedRunner import *
from runner import *
import copy
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time

RUNTIME = 100000
_T = 0
_C = 1  #  C^SW
_D = 2
_ID = 3
_CG = 4 # C^CG
_RSW = 5#
_VD = 6
_RCG = 7

numt, nump, numc = 4, 2, 30


T = np.array([120, 200, 280, 400])
CSW = np.array([30, 50, 70, 100])
DSW = np.array([300, 300, 300, 300])
ID = np.array([1,2,3,4])
CCG = np.array([300, 60, 120, 600])

taskSet = np.array([T, CSW, DSW, ID, CCG]).T
NSW = 2
NCG = 30
NTY = numt 

infinity = 999999

Pn, PSW, PCG =  42.100, 207.581, 13.838
DCG = np.array([infinity, infinity, infinity, infinity])

minP = infinity
n = np.ones(NTY) * infinity
sol = infinity, infinity, n

prevRSW, prevRCG = ID, ID
for NSW in range(1, infinity):
    params = [0.5, 0.5, NTY, NSW, NCG, NUMS]
    newSet = NEWanalysisSW2(taskSet, params, None)
    if np.sum(newSet) == -1:
        continue
    RSW = newSet[:, _RSW]
    if NSW != 1 and np.sum(RSW == prevRSW):
        break
    prevRSW = RSW
    for NCG in range(1, infinity):
        params = [0.5, 0.5, NTY, NSW, NCG, NUMS]
        tempSet = newSet.copy()
        tempSet = np.hstack((tempSet, np.array([DCG]).T))
        newSet2 = NEWanalysisCG3(tempSet, params, None)
        RCG = newSet2[:, _RCG]
        if NCG != 1 and np.sum(RCG == prevRCG):
            break
        prevRCG = RCG
        for x in range(0, NTY):
            n[x] = np.ceil((RCG[x] + RSW[x]) / T[x])
        price = sum(n * Pn) + NSW * PSW + NCG * PCG      
        if price < minP:
            sol = [NSW, NCG, n]
            minP = price
print(sol, minP)

# [2, 11, array([4., 1., 1., 2.])] 1072.58
