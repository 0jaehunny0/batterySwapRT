from matplotlib.pyplot import xlabel, ylabel
import numpy as np
from analysisRTAS import *
from newGenerator import *
# from partitionedRunner import *
from runner import *
import copy
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time


eval3Dir = "evalRTAS/"

RUNTIME = 432000
# RUNTIME = 1000000
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



    return CG / res

def maxmin(a, b):
    data  = pd.DataFrame()
    data["rcg"] = a
    data["rcg2"] = b
    asdf = data.groupby("rcg2").max()
    return (asdf["rcg"]/asdf["rcg"].index).min()

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


        analysisResultRM = RTASanalysisSW2(taskSet, params, batterySet)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet)

            analysisResultCG = RTASanalysisCG2(taskSet, params, batterySet)

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
                res3 = FIFOrunnerAHP1noquasi2(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)
                stationCheck3, chargerCheck3, Preemption3, totalRelease3, totalhighCnt3, acceptCnt3, R_SW_list3, R_CG_list3, realR_SW_list3, realR_CG_list3, RSW_list3, RCG_list3 = res3                              
                res4 = FIFOrunnerAHP22(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
                stationCheck4, chargerCheck4, Preemption4, totalRelease4, totalhighCnt4, acceptCnt4, R_SW_list4, R_CG_list4, realR_SW_list4, realR_CG_list4, RSW_list4, RCG_list4 = res4
                res6 = FIFOrunnerAHP2noquasi2(taskSet, nump, RUNTIME, batterySet, 0, numc, SPORADIC, staticdynamic)                
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
                    maxmin(R_CG_list1, RCG_list1),
                    maxmin(R_CG_list2, RCG_list2),
                    maxmin(R_CG_list3, RCG_list3),
                    maxmin(R_CG_list4, RCG_list4),
                    maxmin(R_CG_list6, RCG_list6),
                ])
                max_RSW_tight_ratio += np.array([
                    maxmin(R_SW_list1, RSW_list1),
                    maxmin(R_SW_list2, RSW_list2),
                    maxmin(R_SW_list3, RSW_list3),
                    maxmin(R_SW_list4, RSW_list4),
                    maxmin(R_SW_list6, RSW_list6),
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

    R_CG_list = R_CG_list / (np.array([CCG]).T) * 1

    resLi = [max_max_RSW_tight_ratio, stationUtil, chargerUtil, R_CG_list, mean_RCG_tight_ratio]
    nameLi = ["max_max_RSW_tight_ratio", "stationUtil", "chargerUtil", "R_CG_list", "mean_RCG_tight_ratio"]


    data = [resLi, nameLi]
    with open(eval3Dir+filename+"5.pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    font = {'family' : 'normal',
            'size'   : 16}

    matplotlib.rc('font', **font)

    for i in range(len(resLi)):
        temp = resLi[i]
        plt.figure()
        plt.plot(varyingLi, temp[:len(varyingLi),0], label="periodic", alpha=0.9, linestyle='-')
        plt.plot(varyingLi, temp[:len(varyingLi),2], label="AHP1 no-quasi", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
        plt.plot(varyingLi, temp[:len(varyingLi),1], label="AHP1", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
        plt.plot(varyingLi, temp[:len(varyingLi),4], label="AHP2 no-quasi", alpha=0.9, linestyle="--", marker="x", markersize=3)
        plt.plot(varyingLi, temp[:len(varyingLi),3], label="AHP2", alpha=0.9, linestyle=":", marker='o', markersize=3)
        plt.xlabel(filename)
        plt.ylabel(nameLi[i])
        # if i ==3:
        #     temp[:len(varyingLi),:].min()
        if i == 0:
            plt.legend()
            plt.ylim(0.95, 1.0)
        plt.tight_layout(pad = 0.1)
        plt.savefig(eval3Dir+filename+"-"+nameLi[i])
    plt.show()



def swapUtil():
    stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # stationUtilLi = [0.7]
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
    pickleSaver("station utilization3", res)
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
    pickleSaver("charger utilization3", res) 
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
    pickleSaver("number of types3", res)
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
    pickleSaver("number of stations3", res) 
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
    pickleSaver("number of chargers3", res) 
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
    pickleSaver("number of batteries3", res) 
    return 1

stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7]
numtLi = np.arange(1,11,1)
numpLi = [1,2,3,4,5]
numcLi = [20, 25, 30, 35, 40]
batLi = np.arange(3,28, 3)

start2 = time.time()

if not os.path.exists(eval3Dir):
    os.makedirs(eval3Dir)

def numT2():
    stationUtilLi = [0.5]
    chargerUtilLi = [0.5]
    numtLi = np.arange(1,11,1)
    numpLi = [2]
    numcLi = [30]
    aux =120

    res = []
    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        aux2 = int(aux/numt)
                        result = mainRunner(params, aux, STATIC)
                        res.append(result)
    res = np.array(res)
    pickleSaver("number of types3", res)
    return 1

# draw("station utilization3", stationUtilLi)
# draw("charger utilization3", chargerUtilLi)
# draw("number of types", numtLi)

swapUtil()
chargerUtil()
# numT2()
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


exit()

with open(eval3Dir+"station utilization2"+".pickle", 'rb') as f:
    data1 = pickle.load(f)


with open(eval3Dir+"charger utilization2"+".pickle", 'rb') as f:
    data2 = pickle.load(f)

resLi1, _ = data1
resLi2, _ = data2


stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7]

temp = resLi1[0]
varyingLi = stationUtilLi
plt.figure()
plt.plot(varyingLi, temp[:len(varyingLi),0], label="periodic", alpha=0.9, linestyle='-')
plt.plot(varyingLi, temp[:len(varyingLi),2], label="AHP1 no-quasi", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
plt.plot(varyingLi, temp[:len(varyingLi),1], label="AHP1", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
plt.plot(varyingLi, temp[:len(varyingLi),4], label="AHP2 no-quasi", alpha=0.9, linestyle="--", marker="x", markersize=3)
plt.plot(varyingLi, temp[:len(varyingLi),3], label="AHP2", alpha=0.9, linestyle=":", marker='o', markersize=3)

temp = resLi2[0]
varyingLi = chargerUtilLi
plt.figure()
plt.plot(varyingLi, temp[:len(varyingLi),0], label="periodic", alpha=0.9, linestyle='-')
plt.plot(varyingLi, temp[:len(varyingLi),2], label="AHP1 no-quasi", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
plt.plot(varyingLi, temp[:len(varyingLi),1], label="AHP1", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
plt.plot(varyingLi, temp[:len(varyingLi),4], label="AHP2 no-quasi", alpha=0.9, linestyle="--", marker="x", markersize=3)
plt.plot(varyingLi, temp[:len(varyingLi),3], label="AHP2", alpha=0.9, linestyle=":", marker='o', markersize=3)

font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[0]
varyingLi = stationUtilLi
ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax1.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax1.set_xlabel(r"$U^{SW}$")
ax1.set_ylabel(r"$RSW\ tightness\ ratio$")
temp = resLi2[0]
varyingLi = chargerUtilLi
ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax2.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax2.set_xlabel(r"$U^{CG}$")
ax1.set_ylim(0.95, 1.0)
f.subplots_adjust(wspace=0)
ax1.legend()
plt.tight_layout(pad = 0.1)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[1]
varyingLi = stationUtilLi
ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax1.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax1.set_xlabel(r"$U^{SW}$")
ax1.set_ylabel(r"$actual\ U^{SW}$(L) and $U^{CG}$(R)")
temp = resLi2[2]
varyingLi = chargerUtilLi
ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax2.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax2.set_xlabel(r"$U^{CG}$")
f.subplots_adjust(wspace=0)
plt.tight_layout(pad = 0.1)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[3]
varyingLi = stationUtilLi
ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax1.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax1.set_xlabel(r"$U^{SW}$")
ax1.set_ylabel(r"$average\ completion\ time$")
temp = resLi2[3]
varyingLi = chargerUtilLi
ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-')
ax2.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":") #, marker ="s", markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle=":", marker='o', markersize=3)
ax2.set_xlabel(r"$U^{CG}$")
f.subplots_adjust(wspace=0)
plt.tight_layout(pad = 0.1)

print("a")



#########################

with open(eval3Dir+"station utilization"+"35.pickle", 'rb') as f:
    data1 = pickle.load(f)


with open(eval3Dir+"charger utilization"+"35.pickle", 'rb') as f:
    data2 = pickle.load(f)

resLi1, _ = data1
resLi2, _ = data2


stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7]

font = {'family' : 'normal',
        'size'   : 23}
matplotlib.rc('font', **font)

from matplotlib import font_manager

font_path = 'C:/Users/jaehunny/Downloads/font/LinBiolinum_Rah.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()

font_dirs = ['C:/Users/jaehunny/Downloads/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

csfont = {'fontname':"Linux Biolinum"}

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[0]
varyingLi = stationUtilLi
ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"PA+BSSM", alpha=0.9, linestyle='-', linewidth=3)
# ax1.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"AA+BSSM-AC", alpha=0.9, linestyle=":", linewidth=3) #, marker ="s", markersize=24)
# ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"AA+BSSM-VA", alpha=0.9, linestyle="--", linewidth=3)
ax1.set_xlabel("swapping machine utilization")
ax1.set_ylabel(r"max RSW tightness ratio")
temp = resLi2[0]
varyingLi = chargerUtilLi
ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-', linewidth=3)
# ax2.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":", linewidth=3) #, marker ="s", markersize=24)
# ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle="--", linewidth=3)
ax2.set_xlabel("charger utilization")
ax1.set_ylim(0.94, 1.0)
f.subplots_adjust(wspace=0)
ax1.legend(prop={'family':"monospace"})
plt.tight_layout(pad = 0.1)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[1]
varyingLi = stationUtilLi
ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"$PA+BSSM$", alpha=0.9, linestyle='-', linewidth=3)
# ax1.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"$AA+BSSM-AC$", alpha=0.9, linestyle=":", linewidth=3) #, marker ="s", markersize=24)
# ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"$AA+BSSM-VA$", alpha=0.9, linestyle="--", linewidth=3)
ax1.set_xlabel("swapping machine utilization")
ax1.set_ylabel("run-time utilization")
ax1.text(0.1,0.65, r"[run-time $U^{SW}$]")
temp = resLi2[2]
varyingLi = chargerUtilLi
ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-', linewidth=3)
# ax2.plot(varyingLi, temp[:len(varyingLi),2], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--") #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle=":", linewidth=3) #, marker ="s", markersize=24)
# ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle="--", linewidth=3)
ax2.set_xlabel("charger utilization")
ax2.text(0.3,0.65, r"[run-time $U^{CG}]$")
f.subplots_adjust(wspace=0)
plt.tight_layout(pad = 0.1)


font = {'family' : 'normal',
        'size'   : 23}
matplotlib.rc('font', **font)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 4.8])
temp = resLi1[4]
varyingLi = stationUtilLi
# ax1.plot(varyingLi, temp[:len(varyingLi),0], label= r"PA+FIFO-W", alpha=0.9, linestyle='-', linewidth=3)
# ax1.plot(varyingLi, temp[:len(varyingLi),1], label=r"PA+FIFO-I", alpha=0.9, linestyle="--", linewidth=3) #, marker = 'x', markersize=24)
ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"FIFO-I", alpha=0.9, linestyle="-", linewidth=3, color="C5")
ax1.plot(varyingLi, temp[:len(varyingLi),3], label=r"FIFO-W", alpha=0.9, linestyle="--", linewidth=3, color="C6") #, marker ="s", markersize=24)
# ax1.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax1.set_xlabel("swapping machine utilization")
ax1.set_ylabel(r"avg RCG tightness ratio")
# ax1.set_yticks([0.5, 0.6, 0.7, 0.8])
temp = resLi2[4]
varyingLi = chargerUtilLi
# ax2.plot(varyingLi, temp[:len(varyingLi),0], label= r"$Periodic$", alpha=0.9, linestyle='-', linewidth=3)
# ax2.plot(varyingLi, temp[:len(varyingLi),1], label=r"BSSM-AC $w/o\ quasi$", alpha=0.9, linestyle="--", linewidth=3) #, marker = 'x', markersize=24)
ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $with\ quasi$", alpha=0.9, linestyle="-", linewidth=3, color="C5")
ax2.plot(varyingLi, temp[:len(varyingLi),3], label=r"BSSM-AC $with\ quasi$", alpha=0.9, linestyle="--", linewidth=3, color="C6") #, marker ="s", markersize=24)
# ax2.plot(varyingLi, temp[:len(varyingLi),4], label=r"BSSM-VA $w/o\ quasi$", alpha=0.9, linestyle="--", marker="x", markersize=3)
ax2.set_xlabel("charger utilization")
f.subplots_adjust(wspace=0)
# ax1.legend(labelspacing=0.05, handlelength=2, fontsize = 20, loc = "upper left", borderpad=0.3)
ax1.legend(loc = "upper left", prop={'family':"monospace"})
plt.tight_layout(pad = 0.1)
plt.show()



resLi1[4][:, 3].sum()  + resLi2[4][:, 3].sum()
7.767902406556737
resLi1[4][:, 4].sum()  + resLi2[4][:, 4].sum()
8.547943134485788

resLi1[4][:, 3].sum()  + resLi2[4][:, 3].sum()
7.0629009747553955
resLi1[4][:, 4].sum()  + resLi2[4][:, 4].sum()
7.865113754926496
"""
resLi[-1][:, 0].mean()+resLi[-1][:, 1].mean()+resLi[-1][:, 3].mean()
2.999219354507089

resLi[-1][:, 2].mean()+resLi[-1][:, 4].mean()
2.1820251653711686

resLi[-1][:5, 0].mean()+resLi[-1][:5, 1].mean()+resLi[-1][:5, 3].mean()
2.9995207678375184

resLi[-1][:5, 2].mean()+resLi[-1][:5, 4].mean()
2.2269459514502055

2.9995207678375184*5 + 2.999219354507089*7
35.992139320737216 / 3
11.997379773579071

2.1820251653711686 * 7 + 2.2269459514502055 * 5
26.408905914849207 / 2
13.204452957424603

1 - 13.204452957424603 / 11.997379773579071
-0.10061140070799279
"""

#    cnt = 0
#    for i in range(0, NUMS):     
#         taskSet = myRes[i, :, :]

#         a = np.sum(NEWanalysisSW2(taskSet, params, batterySet))
#         b = np.sum(RTASanalysisSW(taskSet, params, batterySet))
#         if a!= -1 and b!= -1 and a != b: 
#             print(sum(NEWanalysisSW2(taskSet, params, batterySet)[:, _RSW])/sum(RTASanalysisSW(taskSet, params, batterySet)[:,_RSW]))
#             cnt +=  1