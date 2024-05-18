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

# # Basic parameters
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

DYNAMIC = 1
STATIC = 0

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

def mainRunner(params, AUX, staticdynamic):
    sUtil, cUtil, numt, nump, numc, NUMS= params

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])

    batterySet = np.ones(numt) * AUX
    
    res = 0
    analysisSWFail = 0
    analysisCGFail = 0
    vdli = []

    R_CGdivRCGmean = np.zeros(4)
    realReleasetoRCGend = np.zeros(4)
    R_SWdivRSWmean = np.zeros(4)
    realReleasetoSWend = np.zeros(4)
    PreemptionRatio = np.zeros(4)
    chargerUtil = np.zeros(4)
    stationUtil = np.zeros(4)
    acceptRatio = np.zeros(4)
    
    for i in range(0, NUMS):
        np.random.seed(i)

        taskSet = myRes[i, :, :]

        analysisResultRM = analysisSW(taskSet, params, batterySet)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet)

            analysisResultCG = analysisCG(taskSet, params, batterySet)

            if np.sum(analysisResultCG) != -1:

                start = time.time()

                taskSet = analysisResultCG

                res1 = FIFOrunnerAHP1(taskSet, nump, RUNTIME, batterySet, 0, numc, PERIODIC, staticdynamic)                
                stationCheck1, chargerCheck1, Preemption1, totalRelease1, totalhighCnt1, acceptCnt1, R_SW_list1, R_CG_list1, realR_SW_list1, realR_CG_list1, RSW_list1, RCG_list1 = res1
                res2 = FIFOrunnerAHP1(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)
                stationCheck2, chargerCheck2, Preemption2, totalRelease2, totalhighCnt2, acceptCnt2, R_SW_list2, R_CG_list2, realR_SW_list2, realR_CG_list2, RSW_list2, RCG_list2 = res2                              
                res4 = FIFOrunnerAHP2(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
                stationCheck4, chargerCheck4, Preemption4, totalRelease4, totalhighCnt4, acceptCnt4, R_SW_list4, R_CG_list4, realR_SW_list4, realR_CG_list4, RSW_list4, RCG_list4 = res4
                res6 = FIFOrunnerAHP3(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
                stationCheck6, chargerCheck6, Preemption6, totalRelease6, totalhighCnt6, acceptCnt6, R_SW_list6, R_CG_list6, realR_SW_list6, realR_CG_list6, RSW_list6, RCG_list6 = res6


                stationUtil += np.array([
                    np.sum(stationCheck1 != -1)/RUNTIME/nump,
                    np.sum(stationCheck2 != -1)/RUNTIME/nump,
                    np.sum(stationCheck4 != -1)/RUNTIME/nump,
                    np.sum(stationCheck6 != -1)/RUNTIME/nump
                ])

                chargerUtil += np.array([
                    np.sum(chargerCheck1 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck2 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck4 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck6 != -1)/RUNTIME/numc
                ])

                realReleasetoRCGend +=  np.array([
                    (RCG_list1 +  realR_CG_list1 - R_CG_list1).mean(),
                    (RCG_list2 +  realR_CG_list2 - R_CG_list2).mean(),
                    (RCG_list4 +  realR_CG_list4 - R_CG_list4).mean(),
                    (RCG_list6 +  realR_CG_list6 - R_CG_list6).mean()
                ])

                R_CGdivRCGmean +=  np.array([
                    (R_CG_list1 / RCG_list1).mean(),
                    (R_CG_list2 / RCG_list2).mean(),
                    (R_CG_list4 / RCG_list4).mean(),
                    (R_CG_list6 / RCG_list6).mean()
                ])

                realReleasetoSWend +=  np.array([
                    (RSW_list1 +  realR_SW_list1 - R_SW_list1).mean(),
                    (RSW_list2 +  realR_SW_list2 - R_SW_list2).mean(),
                    (RSW_list4 +  realR_SW_list4 - R_SW_list4).mean(),
                    (RSW_list6 +  realR_SW_list6 - R_SW_list6).mean()
                ])

                R_SWdivRSWmean +=  np.array([
                    (R_SW_list1 / RSW_list1).mean(),
                    (R_SW_list2 / RSW_list2).mean(),
                    (R_SW_list4 / RSW_list4).mean(),
                    (R_SW_list6 / RSW_list6).mean()
                ])

                PreemptionRatio += np.array([
                    Preemption1 / totalRelease1,
                    Preemption2 / totalRelease2,
                    Preemption4 / totalRelease4,
                    Preemption6 / totalRelease6
                ])

                acceptRatio += np.array([
                    acceptCnt1 / totalRelease1,
                    acceptCnt2 / totalRelease2,
                    acceptCnt4 / totalRelease4,
                    acceptCnt6 / totalRelease6
                ])

                end = time.time()
                # print(end - start)


                res += 1
                vdli.append(taskSet[:, _VD].mean())
            else:
                analysisCGFail += 1
        else:
            analysisSWFail += 1

    if res > 0:
        stationUtil = stationUtil / res
        chargerUtil = chargerUtil / res
        realReleasetoRCGend = realReleasetoRCGend / res
        R_CGdivRCGmean = R_CGdivRCGmean / res
        PreemptionRatio = PreemptionRatio / res
        acceptRatio = acceptRatio / res
        realReleasetoSWend = realReleasetoSWend / res
        R_SWdivRSWmean = R_SWdivRSWmean / res
    # return res, vdli, analysisSWFail, analysisCGFail
    return stationUtil, chargerUtil, realReleasetoRCGend, R_SWdivRSWmean, PreemptionRatio, acceptRatio, res, R_CGdivRCGmean, realReleasetoSWend

def draw(filename, varyingLi):


    if not os.path.exists(filename):
        os.makedirs(filename)

    res = pickleLoader(filename)
    res = np.array(res)

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,0])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("station util")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/station util")

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,1])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("charger util")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/charger util")

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,2])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("realReleasetoRCGend")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/realReleasetoRCGend")

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,3])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("R_SWdivRSWmean")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/R_SWdivRSWmean")
    # plt.show()

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,-2])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("R_CGdivRCGmean")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/R_CGdivRCGmean")

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,-1])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("realReleasetoSWend")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/realReleasetoSWend")

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,4])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("PreemptionRatio")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/PreemptionRatio")
    # plt.show()

    plt.figure()
    temp = []
    for i in range(len(varyingLi)):
        temp.append((res[:,5])[i])
    temp = np.array(temp)
    plt.plot(varyingLi, temp[:,0], label="periodic")
    plt.plot(varyingLi, temp[:,1], label="AHP1")
    plt.plot(varyingLi, temp[:,2], label="AHP2")
    plt.plot(varyingLi, temp[:,3], label="AHP3")
    plt.xlabel(filename)
    plt.ylabel("acceptRatio")
    plt.legend()
    plt.tight_layout(pad = 0.2)
    plt.savefig(filename+"/acceptRatio")

    # plt.show()

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
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("station utilization", res)
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
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("charger utilization", res) 
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
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("number of types", res)
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
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("number of stations", res) 
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
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("number of chargers", res) 
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
                            result = mainRunner(params, aux, STATIC)
                            res.append(result)
    res = np.array(res)
    pickleSaver("number of batteries", res) 
    return 1

stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
numtLi = np.arange(1,11,1)
numpLi = [1,2,3,4]
numcLi = [20, 25, 30, 35, 40]
batLi = np.arange(3,37, 3)

start2 = time.time()

swapUtil()
# numBat()
# chargerUtil()
# numT()
# numP()
# numC()


end2 = time.time()

print(end2 - start2)

# draw("station utilization", stationUtilLi)
# draw("charger utilization", chargerUtilLi)
# draw("number of types", numtLi)
# draw("number of stations", numpLi)
# draw("number of chargers", numcLi)
# draw("number of batteries", batLi)

# plt.show()


print("a")