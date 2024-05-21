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


eval3Dir = "eval3/"

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

def readT(params, AUX, staticdynamic):
    sUtil, cUtil, numt, nump, numc, NUMS= params
    name = nameCreator(params)
    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])
    batterySet = np.ones(numt) * AUX
    res = 0
    SW = 0
    CG = 0
    for i in range(0, NUMS):
        np.random.seed(i)
        taskSet = myRes[i, :, :]
        analysisResultRM = NEWanalysisSW2(taskSet, params, batterySet)
        if np.sum(analysisResultRM) != -1:
            taskSet = analysisResultRM
            taskSet = virtualDeadline(taskSet, params, batterySet)
            analysisResultCG = NEWanalysisCG2(taskSet, params, batterySet)
            if np.sum(analysisResultCG) != -1:
                start = time.time()
                taskSet = analysisResultCG
                taskSet = np.array(taskSet, dtype=np.int32)
                res += 1
                SW += np.mean(taskSet[:, _C])
                CG += np.mean(taskSet[:, _CG])

    return CG / res

def mainRunner(params, AUX, staticdynamic):
    sUtil, cUtil, numt, nump, numc, NUMS= params

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])

    batterySet = np.ones(numt) * AUX
    
    res = 0
    analysisSWFail = 0
    analysisCGFail = 0
    vdli = []

    mean_RCG_tight_ratio = np.zeros(5)
    realReleasetoRCGend = np.zeros(5)
    mean_RSW_tight_ratio = np.zeros(5)
    realReleasetoRSWend = np.zeros(5)
    PreemptionRatio = np.zeros(5)
    chargerUtil = np.zeros(5)
    stationUtil = np.zeros(5)
    acceptRatio = np.zeros(5)
    max_RCG_tight_ratio = np.zeros(5)
    max_RSW_tight_ratio = np.zeros(5)
    max_max_RCG_tight_ratio = np.zeros(5)
    max_max_RSW_tight_ratio = np.zeros(5)
    R_CG_list = np.zeros(5)
    R_SW_list = np.zeros(5)
    CSW = 0
    CCG = 0 
    TTT = 0

    
    for i in range(0, NUMS):
        np.random.seed(i)

        taskSet = myRes[i, :, :]

        analysisResultRM = NEWanalysisSW2(taskSet, params, batterySet)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet)

            analysisResultCG = NEWanalysisCG2(taskSet, params, batterySet)

            if np.sum(analysisResultCG) != -1:

                start = time.time()

                taskSet = analysisResultCG

                taskSet = np.array(taskSet, dtype=np.int32)

                CSW += np.mean(taskSet[:, _C])
                CCG += np.mean(taskSet[:, _CG])
                TTT += np.mean(taskSet[:, _T])

                res1 = FIFOrunnerAHP1(taskSet, nump, RUNTIME, batterySet, 0, numc, PERIODIC, staticdynamic)                
                stationCheck1, chargerCheck1, Preemption1, totalRelease1, totalhighCnt1, acceptCnt1, R_SW_list1, R_CG_list1, realR_SW_list1, realR_CG_list1, RSW_list1, RCG_list1 = res1
                res2 = FIFOrunnerAHP1(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)
                stationCheck2, chargerCheck2, Preemption2, totalRelease2, totalhighCnt2, acceptCnt2, R_SW_list2, R_CG_list2, realR_SW_list2, realR_CG_list2, RSW_list2, RCG_list2 = res2                              
                res3 = FIFOrunnerAHP1noquasi(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)
                stationCheck3, chargerCheck3, Preemption3, totalRelease3, totalhighCnt3, acceptCnt3, R_SW_list3, R_CG_list3, realR_SW_list3, realR_CG_list3, RSW_list3, RCG_list3 = res3                              
                res4 = FIFOrunnerAHP2(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
                stationCheck4, chargerCheck4, Preemption4, totalRelease4, totalhighCnt4, acceptCnt4, R_SW_list4, R_CG_list4, realR_SW_list4, realR_CG_list4, RSW_list4, RCG_list4 = res4
                res6 = FIFOrunnerAHP2noquasi(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
                stationCheck6, chargerCheck6, Preemption6, totalRelease6, totalhighCnt6, acceptCnt6, R_SW_list6, R_CG_list6, realR_SW_list6, realR_CG_list6, RSW_list6, RCG_list6 = res6



                stationUtil += np.array([
                    np.sum(stationCheck1 != -1)/RUNTIME/nump,
                    np.sum(stationCheck2 != -1)/RUNTIME/nump,
                    np.sum(stationCheck3 != -1)/RUNTIME/nump,
                    np.sum(stationCheck4 != -1)/RUNTIME/nump,
                    np.sum(stationCheck6 != -1)/RUNTIME/nump,
                
                ])

                chargerUtil += np.array([
                    np.sum(chargerCheck1 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck2 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck3 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck4 != -1)/RUNTIME/numc,
                    np.sum(chargerCheck6 != -1)/RUNTIME/numc,
                    
                ])

                realReleasetoRCGend +=  np.array([
                    (RCG_list1 +  realR_CG_list1 - R_CG_list1).mean(),
                    (RCG_list2 +  realR_CG_list2 - R_CG_list2).mean(),
                    (RCG_list3 +  realR_CG_list3 - R_CG_list3).mean(),
                    (RCG_list4 +  realR_CG_list4 - R_CG_list4).mean(),
                    (RCG_list6 +  realR_CG_list6 - R_CG_list6).mean(),
                    
                ])

                mean_RCG_tight_ratio +=  np.array([
                    (R_CG_list1 / RCG_list1).mean(),
                    (R_CG_list2 / RCG_list2).mean(),
                    (R_CG_list3 / RCG_list3).mean(),
                    (R_CG_list4 / RCG_list4).mean(),
                    (R_CG_list6 / RCG_list6).mean(),
                    
                ])

                realReleasetoRSWend +=  np.array([
                    (RSW_list1 +  realR_SW_list1 - R_SW_list1).mean(),
                    (RSW_list2 +  realR_SW_list2 - R_SW_list2).mean(),
                    (RSW_list3 +  realR_SW_list3 - R_SW_list3).mean(),
                    (RSW_list4 +  realR_SW_list4 - R_SW_list4).mean(),
                    (RSW_list6 +  realR_SW_list6 - R_SW_list6).mean(),
                    
                ])

                mean_RSW_tight_ratio +=  np.array([
                    (R_SW_list1 / RSW_list1).mean(),
                    (R_SW_list2 / RSW_list2).mean(),
                    (R_SW_list3 / RSW_list3).mean(),
                    (R_SW_list4 / RSW_list4).mean(),
                    (R_SW_list6 / RSW_list6).mean(),
                    
                ])

                PreemptionRatio += np.array([
                    Preemption1 / totalRelease1,
                    Preemption2 / totalRelease2,
                    Preemption3 / totalRelease3,
                    Preemption4 / totalRelease4,
                    Preemption6 / totalRelease6,
                    
                ])

                acceptRatio += np.array([
                    acceptCnt1 / totalRelease1,
                    acceptCnt2 / totalRelease2,
                    acceptCnt3 / totalRelease3,
                    acceptCnt4 / totalRelease4,
                    acceptCnt6 / totalRelease6,
                    
                ])

                max_RCG_tight_ratio += np.array([
                    (R_CG_list1 / RCG_list1).max(),
                    (R_CG_list2 / RCG_list2).max(),
                    (R_CG_list3 / RCG_list3).max(),
                    (R_CG_list4 / RCG_list4).max(),
                    (R_CG_list6 / RCG_list6).max(),
                ])
                max_RSW_tight_ratio += np.array([
                    (R_SW_list1 / RSW_list1).max(),
                    (R_SW_list2 / RSW_list2).max(),
                    (R_SW_list3 / RSW_list3).max(),
                    (R_SW_list4 / RSW_list4).max(),
                    (R_SW_list6 / RSW_list6).max(),
                ])

                max_max_RCG_tight_ratio = np.fmax(max_max_RCG_tight_ratio, np.array([
                    (R_CG_list1 / RCG_list1).max(),
                    (R_CG_list2 / RCG_list2).max(),
                    (R_CG_list3 / RCG_list3).max(),
                    (R_CG_list4 / RCG_list4).max(),
                    (R_CG_list6 / RCG_list6).max(),
                ]))
                max_max_RSW_tight_ratio = np.fmax(max_max_RSW_tight_ratio, np.array([
                    (R_SW_list1 / RSW_list1).max(),
                    (R_SW_list2 / RSW_list2).max(),
                    (R_SW_list3 / RSW_list3).max(),
                    (R_SW_list4 / RSW_list4).max(),
                    (R_SW_list6 / RSW_list6).max(),
                ]))

                R_CG_list += np.array([
                    R_CG_list1.mean(),
                    R_CG_list2.mean(),
                    R_CG_list3.mean(),
                    R_CG_list4.mean(),
                    R_CG_list6.mean(),
                ])

                R_SW_list += np.array([
                    R_SW_list1.mean(),
                    R_SW_list2.mean(),
                    R_SW_list3.mean(),
                    R_SW_list4.mean(),
                    R_SW_list6.mean(),
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
        realReleasetoRSWend = realReleasetoRSWend / res
        mean_RCG_tight_ratio = mean_RCG_tight_ratio / res
        mean_RSW_tight_ratio = mean_RSW_tight_ratio / res
        max_RCG_tight_ratio = max_RCG_tight_ratio / res
        max_RSW_tight_ratio = max_RSW_tight_ratio / res
        PreemptionRatio = PreemptionRatio / res
        acceptRatio = acceptRatio / res
        R_CG_list = R_CG_list / res
        R_SW_list = R_SW_list / res
        CSW /= res
        CCG /= res
        TTT /= res
    # return res, vdli, analysisSWFail, analysisCGFail
    return stationUtil, chargerUtil, realReleasetoRCGend, realReleasetoRSWend, mean_RCG_tight_ratio, mean_RSW_tight_ratio, max_RCG_tight_ratio, max_RSW_tight_ratio, max_max_RCG_tight_ratio, max_max_RSW_tight_ratio, PreemptionRatio, acceptRatio, res, R_CG_list, R_SW_list, CSW, CCG, TTT

def draw(filename, varyingLi):


    if not os.path.exists(filename):
        os.makedirs(filename)

    res = pickleLoader(filename)
    res = np.array(res)

    res = np.stack(res)
    stationUtil, chargerUtil, realReleasetoRCGend, realReleasetoRSWend, mean_RCG_tight_ratio, mean_RSW_tight_ratio, max_RCG_tight_ratio, max_RSW_tight_ratio, max_max_RCG_tight_ratio, max_max_RSW_tight_ratio, PreemptionRatio, acceptRatio, success, R_CG_list, R_SW_list, CSW, CCG, TTT = res.T
    stationUtil = np.stack(stationUtil)
    chargerUtil = np.stack(chargerUtil)
    realReleasetoRCGend = np.stack(realReleasetoRCGend)
    realReleasetoRSWend = np.stack(realReleasetoRSWend)
    mean_RCG_tight_ratio = np.stack(mean_RCG_tight_ratio)
    mean_RSW_tight_ratio = np.stack(mean_RSW_tight_ratio)
    max_RCG_tight_ratio = np.stack(max_RCG_tight_ratio)
    max_RSW_tight_ratio = np.stack(max_RSW_tight_ratio)
    max_max_RCG_tight_ratio = np.stack(max_max_RCG_tight_ratio)
    max_max_RSW_tight_ratio = np.stack(max_max_RSW_tight_ratio)
    PreemptionRatio = np.stack(PreemptionRatio)
    acceptRatio = np.stack(acceptRatio)
    R_CG_list = np.stack(R_CG_list) 
    R_SW_list = np.stack(R_SW_list)

    CSW[CSW == 0] = 1
    CCG[CCG == 0] = 1
    TTT[TTT == 0] = 1

    resLi = [stationUtil, chargerUtil, mean_RCG_tight_ratio, mean_RSW_tight_ratio, max_RCG_tight_ratio, max_RSW_tight_ratio, max_max_RCG_tight_ratio, max_max_RSW_tight_ratio, R_CG_list, R_SW_list]
    nameLi = ["stationUtil", "chargerUtil", "mean_RCG_tight_ratio", "mean_RSW_tight_ratio", "max_RCG_tight_ratio", "max_RSW_tight_ratio", "max_max_RCG_tight_ratio", "max_max_RSW_tight_ratio", "R_CG_list", "R_SW_list"]


    resLi = [stationUtil, chargerUtil, R_CG_list, max_max_RSW_tight_ratio]
    nameLi = ["stationUtil", "chargerUtil", "R_CG_list", "max RSW  tight ratio"]

    R_CG_list = R_CG_list / (np.array([CCG]).T) * 100

    np.array([63.66491841, 64.86652794, 64.85897436, 64.92789072, 65.50133869,66.97458791, 69.49890988])
    np.array([4467.61509324, 2299.02110583, 1532.6971917 , 1155.09942821, 938.15930388,  808.00137363,  729.25290698])

    R_SW_list = R_SW_list / np.array([[63.66491841, 64.86652794, 64.85897436, 64.92789072, 65.50133869,66.97458791, 69.49890988]]).T * 100
    R_CG_list = R_CG_list / np.array([[4467.61509324, 2299.02110583, 1532.6971917 , 1155.09942821, 938.15930388,  808.00137363,  729.25290698]]).T * 100
    
    np.array([75.15681363, 67.93399168, 65.50133869, 65.85334347, 68.39473684, 1, 1])
    np.array([ 677.37650301,  778.57328482,  938.15930388, 1155.58662614, 1438.19736842])
    # resLi = [stationUtil, chargerUtil, realReleasetoRCGend, realReleasetoRSWend, mean_RCG_tight_ratio, mean_RSW_tight_ratio, max_RCG_tight_ratio, max_RSW_tight_ratio, max_max_RCG_tight_ratio, max_max_RSW_tight_ratio, PreemptionRatio, acceptRatio, R_CG_list, R_SW_list]
    # nameLi = ["stationUtil", "chargerUtil", "realReleasetoRCGend", "realReleasetoRSWend", "mean_RCG_tight_ratio", "mean_RSW_tight_ratio", "max_RCG_tight_ratio", "max_RSW_tight_ratio", "max_max_RCG_tight_ratio", "max_max_RSW_tight_ratio", "PreemptionRatio", "acceptRatio", "R_CG_list", "R_SW_list"]

    # R_SW_list = R_SW_list / np.array([[75.15681363, 67.93399168, 65.50133869, 65.85334347, 68.39473684, 1, 1]]).T * 100
    # R_CG_list = R_CG_list / np.array([[677.37650301,  778.57328482,  938.15930388, 1155.58662614, 1438.19736842, 1, 1]]).T * 100
   


    resLi = [stationUtil, chargerUtil, R_CG_list, R_SW_list]
    nameLi = ["stationUtil", "chargerUtil", "R_CG_list", "R_SW_list"]


    for i in range(len(resLi)):
        temp = resLi[i]
        plt.figure()
        plt.plot(varyingLi, temp[:,0], label="periodic", alpha=0.5)
        plt.plot(varyingLi, temp[:,1], label="AHP1", alpha=0.5)
        plt.plot(varyingLi, temp[:,3], label="AHP2 no-quasi", alpha=0.5)
        plt.plot(varyingLi, temp[:,2], label="AHP2", alpha=0.5)
        plt.xlabel(filename)
        plt.ylabel(nameLi[i])
        plt.legend()
        plt.tight_layout(pad = 0.2)
        plt.savefig(eval3Dir+filename+"-"+nameLi[i])
    plt.show()



def swapUtil():
    stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [2]
    numcLi = [30]
    aux =12
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
    aux =12

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
    aux =12

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
    numpLi = [1,2,3,4,5]
    numcLi = [30]
    aux =12
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
    aux =12

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
    batLi = np.arange(3,28, 3)

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
numpLi = [1,2,3,4,5]
numcLi = [20, 25, 30, 35, 40]
batLi = np.arange(3,28, 3)

start2 = time.time()

if not os.path.exists(eval3Dir):
    os.makedirs(eval3Dir)

# draw("station utilization", stationUtilLi)
draw("charger utilization", chargerUtilLi)
# draw("number of types", numtLi)

# swapUtil()
# chargerUtil()
# numT()
# numP()
# numC()
# numBat()



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