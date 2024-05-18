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

# camelCase
n, util, nSets = None, None, None

def loadByName(name):
    return pickleLoader(name)

def taskSetLoader(params):
    result = taskSetsGenerator(params)
    return result

def TtoC(taskSet, targetUtil):
    T = taskSet[:, _T]

    C = np.ones(T.shape[0], dtype=np.int32)

    Tlen = T.shape[0]

    lastUtil = sum(C/T)
    lastIdx = -1
    while True:
        if sum(C/T) >= targetUtil:
            if lastIdx != -1:
                if abs(targetUtil - sum(C/T)) < abs(targetUtil - lastUtil):
                    C[lastIdx] -= 1
            break
        idx = np.random.randint(0, Tlen)

        if C[idx] < T[idx]:
            lastUtil = sum(C/T)
            C[idx] += 1
            lastIdx = idx

    return T, C

def mainRunner(params, numc, AUX):

    sUtil, cUtil, numt, nump, numc, NUMS= params

    name = nameCreator(params)
    myRes = loadByName(name) 
    batterySet = np.ones(numt) * AUX

    res = 0
    analysisSWFail = 0
    analysisCGFail = 0
    vdli = []

    
    for i in range(0, NUMS):
        np.random.seed(i)
        taskSet = myRes[i, :, :]
        analysisResultRM = analysisSW(taskSet, params, batterySet)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet)

            analysisResultCG = analysisCG(taskSet, params, batterySet)

            if np.sum(analysisResultCG) != -1:
                res += 1
                vdli.append(taskSet[:, _VD].mean())
            else:
                analysisCGFail += 1
        else:
            analysisSWFail += 1

    return res, vdli, analysisSWFail, analysisCGFail

def swapUtil():
    stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [2]
    numcLi = [30]
    aux =15

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = mainRunner(params, False, aux)
                        res.append(result)
    plt.figure()
    plt.plot(stationUtilLi, np.array(res)[:,0]/10)
    plt.xlabel("station utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()
    return 1

def chargerUtil():
    stationUtilLi = [0.5]
    chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    numtLi = [4]
    numpLi = [2]
    numcLi = [30]
    aux =15

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = mainRunner(params, False, aux)
                        res.append(result)
    plt.figure()
    plt.plot(chargerUtilLi, np.array(res)[:,0]/10)
    plt.xlabel("charger utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

    return 1

def numT():
    stationUtilLi = [0.5]
    chargerUtilLi = [0.5]
    numtLi = np.arange(1,11,1)
    numpLi = [2]
    numcLi = [30]
    aux =15

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = mainRunner(params, False, aux)
                        res.append(result)

    plt.figure()
    plt.plot(numtLi, np.array(res)[:,0]/10)
    plt.xlabel("number of types")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

    return 1


def numP():
    stationUtilLi = [0.5]
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [1,2,3,4]
    numcLi = [30]
    aux =15

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = mainRunner(params, False, aux)
                        res.append(result)

    plt.figure()
    plt.plot(numpLi, np.array(res)[:,0]/10)
    plt.xlabel("number of stations")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

    return 1


def numC():
    stationUtilLi = [0.5]
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [2]
    numcLi = [20, 25, 30, 35, 40]
    aux =15

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = mainRunner(params, False, aux)
                        res.append(result)

    plt.figure()
    plt.plot(numcLi, np.array(res)[:,0]/10)
    plt.xlabel("number of chargers")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

    return 1


def numBat():
    batLi = np.arange(3,37, 3)

    stationUtilLi = [0.5]
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [2]
    numcLi = [30]

    res = []

    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        for aux in batLi:
                            params = [sUtil, cUtil, numt, nump, numc, NUMS]
                            result = mainRunner(params, False, aux)
                            res.append(result)
    plt.figure()
    plt.plot(batLi, np.array(res)[:,0]/10)
    plt.xlabel("number of batteries")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()


numBat()
chargerUtil()
swapUtil()
numT()
numP()
numC()


plt.show()




print("a")