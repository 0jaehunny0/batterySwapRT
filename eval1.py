from matplotlib.pyplot import xlabel, ylabel
import numpy as np
from analysis import *
from generator import *
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
_CG = 8
_CG = 4 # C^CG

# Basic parameters
UTIL = 0.75 # util  # (0-1] target utilization
NUMT = 3    # n     # [1- ] number of tasks / number of car types
NUMP = 4    # m     # [1- ] number of processors / number of swap stations
NUMS = 1000 # nSets # [1- ] number of sets / number of scenarios
MINT = 1    # minT  # [1- ] minimum periods 
MAXT = 100  # maxT  # [1- ] maximum periods
MIND = 1  # minD  # [1- ] minimum deadline (multiple of WCET)
MAXD = 5    # maxD  # [0- ] maximum deadline (multiple of periods)

# Optional parameters
OPTS = 1    #      # [0- ] random seed value
OPTD = 1    #      # [0,1] 0: implicit-deadlines, 1: constrained-deadlines 

# Battery swap station-specific parameters
NUMC = 5    # cg    # [1- ] number of chargers

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

def mainRunner(params, chargerNUM, change, AUX):
    assert len(params) == 10, "parameters: UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD"
    global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD
    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD = params

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])

    # batterySet = np.random.randint(10, 100, NUMT)
    batterySet = np.ones(NUMT) * AUX
    # C_CG = np.random.randint(1, 5, NUMT)
    
    res = 0
    analysisSWFail = 0
    analysisCGFail = 0
    vdli = []

    analysisFail1, analysisFail2 = 0, 0
    # for i in range(0, NUMS):
    for i in range(0, NUMS):
        np.random.seed(i)
    # for i in range(794, NUMS):

        taskSet = myRes[i, :, :]

        targetUtil = 0.3 * chargerNUM
        _, C_CG = TtoC(taskSet, targetUtil)

        if change:
            # change C with C_CG
            XXX = taskSet[:, _C].copy()
            taskSet[:, _C] = C_CG
            C_CG = XXX


        taskSet = np.hstack((taskSet, np.array([C_CG]).T))

        analysisResultRM = analysisSW(taskSet, params, batterySet, C_CG, chargerNUM)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet, C_CG, chargerNUM)

            analysisResultCG = analysisCG(taskSet, params, batterySet, C_CG, chargerNUM)

            if np.sum(analysisResultCG) != -1:

                res += 1
                vdli.append(taskSet[:, _VD].mean())
            else:
                analysisCGFail += 1
        else:
            analysisSWFail += 1

    return res, vdli, analysisSWFail, analysisCGFail

def swapUtil():
    utilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    numtLi = [4]
    numpLi = [4]
    numcLi = [4]

    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc, False, 10)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])
    
    plt.figure()
    plt.plot(utilLi, np.array(res)/10)
    plt.xlabel("swap utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()
    return 1

def chargerUtil():
    utilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    numtLi = [4]
    numpLi = [4]
    numcLi = [4]

    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc, True, 10)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(utilLi, np.array(res)/10)
    plt.xlabel("charger utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()

    return 1

def numT():
    utilLi = [0.3]
    numtLi = [2,4,6,8,10,12,14,16,18,20]
    numpLi = [4]
    numcLi = [4]

    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc, False, 10)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numtLi, np.array(res)/10)
    plt.xlabel("number of types")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()

    return 1


def numP():
    utilLi = [0.3]
    numtLi = [4]
    numpLi = np.arange(1,11,1)
    numcLi = [4]

    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc, False, 10)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numpLi, np.array(res)/10)
    plt.xlabel("number of stations")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()

    return 1


def numC():
    utilLi = [0.3]
    numtLi = [4]
    numpLi = [4]
    numcLi = np.arange(1,11,1)

    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc, False, 10)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numcLi, np.array(res)/10)
    plt.xlabel("number of chargers")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()

    return 1


def numBat():

    utilLi = [0.3]
    numtLi = [4]
    numpLi = [4]
    numcLi = [4]

    batLi = np.arange(1,21, 1)


    res = []
    res2 = []
    analysisSWFailLi = []
    analysisCGFailLi = []

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    for aux in batLi:
                        UTIL, NUMT, NUMP = util, numt, nump
                        params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                        result = mainRunner(params, numc, False, aux)
                        res.append(result[0])
                        res2.append(np.array(result[1]).mean())
                        analysisSWFailLi.append(result[2])
                        analysisCGFailLi.append(result[3])
    plt.figure()
    plt.plot(batLi, np.array(res)/10)
    plt.xlabel("number of batteries")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    plt.show()

if __name__ == "__main__":

    swapUtil()
    chargerUtil()
    numT()
    numBat()
    numP()
    numC()


    


    

    print("a")