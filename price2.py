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

# Basic parameters
# UTIL = 0.75 # util  # (0-1] target utilization
# NUMT = 3    # n     # [1- ] number of tasks / number of car types
# NUMP = 4    # m     # [1- ] number of processors / number of swap stations
NUMS = 1000 # nSets # [1- ] number of sets / number of scenarios
# MINT = 1    # minT  # [1- ] minimum periods 
# MAXT = 100  # maxT  # [1- ] maximum periods
# MIND = 1  # minD  # [1- ] minimum deadline (multiple of WCET)
# MAXD = 5    # maxD  # [0- ] maximum deadline (multiple of periods)

# # Optional parameters
# OPTS = 1    #      # [0- ] random seed value
# OPTD = 1    #      # [0,1] 0: implicit-deadlines, 1: constrained-deadlines 

# # Battery swap station-specific parameters
# NUMC = 5    # cg    # [1- ] number of chargers

PERIODIC = 0
SPORADIC = 1

eva1Dir = "eval/"

def pickleSaver2(name, data): # save data
    if not os.path.exists(eva1Dir):
        os.makedirs(eva1Dir)
    with open(eva1Dir+name+".pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def pickleLoader2(name): # load data
    with open(eva1Dir+name+".pickle", 'rb') as f:
        data = pickle.load(f)
    return data

# camelCase
n, util, nSets = None, None, None

def loadByName(name):
    return pickleLoader(name)

def taskSetLoader(params):
    result = taskSetsGenerator(params)
    return result

def mainRunner(params, numc, AUX):

    sUtil, cUtil, numt, nump, numc, NUMS= params

    name = nameCreator(params)
    myRes = loadByName(name) 
    batterySet = np.ones(numt) * AUX

    maxVD = 0

    totalminP = 0
    cnt = 0

    NTY = numt 
    infinity = 9999999 # maxVD = 6465970.0
    for i in range(0, NUMS):
        np.random.seed(i)
        taskSet = myRes[i, :, :]

        flag = True
        SW3 = NEWanalysisSW2(taskSet, params, batterySet)
        if np.sum(SW3) != -1:
            VD3 = virtualDeadline(SW3, params, batterySet)
            CG3 = NEWanalysisCG2(VD3, params, batterySet)
            if np.sum(CG3) != -1:
                cnt += 1
                flag = False
        if flag: continue

        T2 = taskSet[:,_T]
        CSW2 = taskSet[:,_C]
        DSW2 = taskSet[:,_D]
        ID2 = taskSet[:,_ID]
        CCG2 = taskSet[:,_CG]
        NSW = 2
        NCG = 30
        taskSet2 = np.array([T2, CSW2, DSW2, ID2, CCG2]).T
        taskSet2 = np.array(taskSet2, dtype=np.int32)

        # Pn, PSW, PCG =  42.100, 207.581, 13.838
        Pn, PSW, PCG =  42, 208, 14

        DCG = np.array([infinity, infinity, infinity, infinity])

        minP = infinity
        n = np.ones(NTY) * infinity
        sol = infinity, infinity, n

        prevRSW, prevRCG = ID2, ID2
        for NSW in range(2, infinity):
            params2 = np.array([0.5, 0.5, NTY, NSW, NCG, NUMS])
            newSet = NEWanalysisSW3(taskSet2, params2, None)
            if newSet[0][0] == 0:
                continue
            RSW = newSet[:, _RSW]
            if NSW != 1 and np.sum(RSW == prevRSW):
                break
            prevRSW = RSW
            for NCG in range(31, infinity):
                params2 = np.array([0.5, 0.5, NTY, NSW, NCG, NUMS])
                tempSet = newSet.copy()
                tempSet = np.hstack((tempSet, np.array([DCG]).T))
                tempSet = np.array(tempSet, dtype=np.int32)
                newSet2 = NEWanalysisCG4(tempSet, params2, None)
                RCG = newSet2[:, _RCG]
                if NCG != 1 and np.sum(RCG == prevRCG):
                    break
                prevRCG = RCG
                for x in range(0, NTY):
                    n[x] = np.ceil((RCG[x] + RSW[x]) / T2[x])
                price = sum(n * Pn) + NSW * PSW + NCG * PCG   
                if price < minP:
                    sol = [NSW, NCG, n.copy()]
                    minP = price
        print(sol, minP, i)

        totalminP += minP

        # if sol[0] == 3: 
        #     print(1)
        # SW3 = NEWanalysisSW2(taskSet, params, batterySet)
        # if np.sum(SW3) != -1:

        #     VD3 = virtualDeadline(SW3, params, batterySet)
        #     # CG3 = NEWanalysisCG2(VD3, params, batterySet)

        #     if max(VD3[:,_VD]) > maxVD:
        #         maxVD = max(VD3[:,_VD])

    print(totalminP, cnt)
    return 0


sUtil = 0.5
cUtil = 0.5
numt = 4
nump = 2
numc = 30




params = [sUtil, cUtil, numt, nump, numc, NUMS]
result = mainRunner(params, False, 12)

Pn, PSW, PCG =  42, 208, 14


font = {'family' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)




plt.figure()
PSWW = np.ones(7) * np.arange(2,9,1)  * PSW
PCGG = np.ones(7) * np.array([30, 29, 31, 28, 28, 30, 30]) * PCG
minP = np.array([1886,2080,2274,2482,2690,2884,3092])
Pnn = minP - PSWW - PCGG

x = np.arange(2,9,1)
plt.bar(x, minP, label= r'$P^{SW}$')
plt.bar(x, Pnn+PCGG, label= r'$P^{CG}$')
plt.bar(x, Pnn, label= r'$P^{n}$')
plt.legend()
plt.xlabel("NSW")
plt.hlines(min(minP),1.5,8.5, colors="red", linestyles="--")
plt.ylabel("Minimum cost ($)")
plt.tight_layout(pad=0.1)

plt.figure()
PSWW = np.ones(7) * 2 * PSW
PCGG = np.ones(7) * np.arange(27,34,1) * PCG
minP = np.array([1970, 1984, 1956, 1886, 1900, 1914, 1928])
Pnn = minP - PSWW - PCGG

x = np.arange(27,34,1)
plt.bar(x, minP, label= r'$P^{SW}$')
plt.bar(x, Pnn+PCGG, label= r'$P^{CG}$')
plt.bar(x, Pnn, label= r'$P^{n}$')
# plt.legend()
plt.xlabel("NCG")
plt.ylabel("Minimum cost ($)")
plt.hlines(min(minP),26.5,33.5, colors="red", linestyles="--")
plt.tight_layout(pad=0.1)


#########################################################################




plt.figure()
PSWW = np.ones(11) * np.arange(2,13,1)  * PSW
PCGG = np.ones(11) * np.array([30, 29, 31, 28, 28, 30, 30, 30, 30, 30, 30]) * PCG
minP = np.array([1886,2080,2274,2482,2690,2884,3092,3300,3508,3716,3924])
Pnn = minP - PSWW - PCGG

x = np.arange(2,13,1)
plt.bar(x, minP, label= r'$P^{SW}$')
plt.bar(x, Pnn+PCGG, label= r'$P^{CG}$')
plt.bar(x, Pnn, label= r'$P^{n}$')
plt.legend()
plt.xlabel("NSW")
plt.hlines(min(minP),1.5,12.5, colors="red", linestyles="--")
plt.ylabel("Minimum cost ($)")
plt.tight_layout(pad=0.1)

plt.figure()
PSWW = np.ones(11) * 2 * PSW
PCGG = np.ones(11) * np.arange(25,36,1) * PCG
minP = np.array([1984, 1956, 1970, 1984, 1956, 1886, 1900, 1914, 1928, 1942, 1914])
Pnn = minP - PSWW - PCGG

x = np.arange(25,36,1)
plt.bar(x, minP, label= r'$P^{SW}$')
plt.bar(x, Pnn+PCGG, label= r'$P^{CG}$')
plt.bar(x, Pnn, label= r'$P^{n}$')
# plt.legend()
plt.xlabel("NCG")
plt.ylabel("Minimum cost ($)")
plt.hlines(min(minP),24.5,35.5, colors="red", linestyles="--")
plt.tight_layout(pad=0.1)




"""
[1] non sched
[2, 25, np.array([10.,  9.,  6.,  4.])] 1984.0 0
[2, 26, np.array([9., 9., 6., 4.])] 1956.0 0
[2, 27, np.array([9., 9., 6., 4.])] 1970.0 0
[2, 28, np.array([9., 9., 6., 4.])] 1984.0 0
[2, 29, np.array([9., 9., 5., 4.])] 1956.0 0
[2, 30, np.array([9., 8., 5., 3.])] 1886.0 0
[2, 31, np.array([9., 8., 5., 3.])] 1900.0 0
[2, 32, np.array([9., 8., 5., 3.])] 1914.0 0
[2, 33, np.array([9., 8., 5., 3.])] 1928.0 0
[2, 34, np.array([9., 8., 5., 3.])] 1942.0 0
[2, 35, np.array([8., 8., 5., 3.])] 1914.0 0

(9999999, 9999999, array([9999999., 9999999., 9999999., 9999999.])) 9999999 0
[2, 30, np.array([9., 8., 5., 3.])] 1886.0 0
[3, 29, np.array([9., 8., 5., 3.])] 2080.0 0
[4, 31, np.array([8., 8., 5., 3.])] 2274.0 0
[5, 28, np.array([9., 8., 5., 3.])] 2482.0 0
[6, 28, np.array([9., 8., 5., 3.])] 2690.0 0
[7, 30, np.array([8., 8., 5., 3.])] 2884.0 0
[8, 30, np.array([8., 8., 5., 3.])] 3092.0 0
[9, 30, np.array([8., 8., 5., 3.])] 3300.0 0
[10, 30, np.array([8., 8., 5., 3.])] 3508.0 0
[11, 30, np.array([8., 8., 5., 3.])] 3716.0 0
[12, 30, np.array([8., 8., 5., 3.])] 3924.0 0
"""

"""

Pn, PSW, PCG =  42, 208, 14

1433510.0 747

747 * (Pn * 4 * 12 + PSW * 2 + PCG * 30)

"""
