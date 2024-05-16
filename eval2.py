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

    if change:
        CUTIL = UTIL
        UTIL = 0.3

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])

    batterySet = np.ones(NUMT) * AUX
    
    res = 0
    analysisSWFail = 0
    analysisCGFail = 0
    vdli = []

    PreemptionRatio = np.zeros(6)
    RealRCGmean = np.zeros(6)
    realRCGdivRCGmean = np.zeros(6)
    RCGdivRCGmean = np.zeros(6)
    realRCGdivRCGmean = np.zeros(6)
    RCGdivRCGmean = np.zeros(6)
    chargerUtil = np.zeros(6)
    stationUtil = np.zeros(6)
    
    for i in range(0, NUMS):
        np.random.seed(i)

        taskSet = myRes[i, :, :]

        if change:
            targetUtil = CUTIL * chargerNUM
        else:
            targetUtil = 0.3 * chargerNUM
        _, C_CG = TtoC(taskSet, targetUtil)

        taskSet = np.hstack((taskSet, np.array([C_CG]).T))

        analysisResultRM = analysisSW(taskSet, params, batterySet, C_CG, chargerNUM)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet, C_CG, chargerNUM)

            analysisResultCG = analysisCG(taskSet, params, batterySet, C_CG, chargerNUM)

            if np.sum(analysisResultCG) != -1:

                taskSet = analysisResultCG

                res1 = FIFOrunnerAHP1(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                stationCheck1, chargerCheck1, Preemption1, totalRelease1, totalhighCnt1, acceptCnt1, R_SW_list1, R_CG_list1, realR_SW_list1, realR_CG_list1, RSW_list1, RCG_list1 = res1
                res2 = FIFOrunnerAHP1(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)
                stationCheck2, chargerCheck2, Preemption2, totalRelease2, totalhighCnt2, acceptCnt2, R_SW_list2, R_CG_list2, realR_SW_list2, realR_CG_list2, RSW_list2, RCG_list2 = res2                              
                res3 = FIFOrunnerAHP2(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)               
                stationCheck3, chargerCheck3, Preemption3, totalRelease3, totalhighCnt3, acceptCnt3, R_SW_list3, R_CG_list3, realR_SW_list3, realR_CG_list3, RSW_list3, RCG_list3 = res3 
                res4 = FIFOrunnerAHP2(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                stationCheck4, chargerCheck4, Preemption4, totalRelease4, totalhighCnt4, acceptCnt4, R_SW_list4, R_CG_list4, realR_SW_list4, realR_CG_list4, RSW_list4, RCG_list4 = res4
                res5 = FIFOrunnerAHP3(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                stationCheck5, chargerCheck5, Preemption5, totalRelease5, totalhighCnt5, acceptCnt5, R_SW_list5, R_CG_list5, realR_SW_list5, realR_CG_list5, RSW_list5, RCG_list5 = res5
                res6 = FIFOrunnerAHP3(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                stationCheck6, chargerCheck6, Preemption6, totalRelease6, totalhighCnt6, acceptCnt6, R_SW_list6, R_CG_list6, realR_SW_list6, realR_CG_list6, RSW_list6, RCG_list6 = res6


                chargerUtil += np.array([
                    np.sum(chargerCheck1 != -1)/RUNTIME/chargerNUM,
                    np.sum(chargerCheck2 != -1)/RUNTIME/chargerNUM,
                    np.sum(chargerCheck3 != -1)/RUNTIME/chargerNUM,
                    np.sum(chargerCheck4 != -1)/RUNTIME/chargerNUM,
                    np.sum(chargerCheck5 != -1)/RUNTIME/chargerNUM,
                    np.sum(chargerCheck6 != -1)/RUNTIME/chargerNUM
                ])

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
                    result = mainRunner(params, numc, False, 5)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])
    
    plt.figure()
    plt.plot(utilLi, np.array(res)/10)
    plt.xlabel("swap utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()
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
                    result = mainRunner(params, numc, True, 5)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(utilLi, np.array(res)/10)
    plt.xlabel("charger utilization")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

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
                    result = mainRunner(params, numc, False, 5)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numtLi, np.array(res)/10)
    plt.xlabel("number of types")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

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
                    result = mainRunner(params, numc, False, 5)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numpLi, np.array(res)/10)
    plt.xlabel("number of stations")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

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
                    result = mainRunner(params, numc, False, 5)
                    res.append(result[0])
                    res2.append(np.array(result[1]).mean())
                    analysisSWFailLi.append(result[2])
                    analysisCGFailLi.append(result[3])

    plt.figure()
    plt.plot(numcLi, np.array(res)/10)
    plt.xlabel("number of chargers")
    plt.ylabel("analysis pass ratio")
    plt.tight_layout(pad = 0.2)
    # plt.show()

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
    # plt.show()



chargerUtil()
numBat()
swapUtil()
numT()
numP()
numC()


plt.show()




print("a")