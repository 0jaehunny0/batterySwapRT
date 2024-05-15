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

def mainRunner(params, chargerNUM):
    assert len(params) == 10, "parameters: UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD"
    global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD
    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD = params

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD, OPTG, OPTP])

    np.random.seed(0)
    batterySet = np.random.randint(10, 100, NUMT)
    C_CG = np.random.randint(1, 5, NUMT)
    
    analysisFail1, analysisFail2 = 0, 0
    for i in range(0, NUMS):
    # for i in range(794, NUMS):

        taskSet = myRes[i, :, :]

        taskSet = np.hstack((taskSet, np.array([C_CG]).T))

        analysisResultRM = analysisSW(taskSet, params, batterySet, C_CG, chargerNUM)

        if np.sum(analysisResultRM) != -1:

            taskSet = analysisResultRM

            taskSet = virtualDeadline(taskSet, params, batterySet, C_CG, chargerNUM)

            analysisResultCG = analysisCG(taskSet, params, batterySet, C_CG, chargerNUM)

            if np.sum(analysisResultCG) != -1:

                taskSet = analysisResultCG

                res1 = FIFOrunnerAHP2(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                res2 = FIFOrunnerAHP2(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                              
                res3 = FIFOrunnerAHP3(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                res4 = FIFOrunnerAHP3(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                res5 = FIFOrunnerAHP4(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                res6 = FIFOrunnerAHP4(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                
                # res1 = FIFOrunner(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                # res2 = FIFOrunner(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                # res3 = FIFOrunner_dynamic(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, PERIODIC)                
                # res4 = FIFOrunner_dynamic(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, SPORADIC)                
                
                # resUtil1 = sum(sum(res1[0] != -1))/RUNTIME/NUMP
                # resUtil2 = sum(sum(res2[0] != -1))/RUNTIME/NUMP
                # resUtil3 = sum(sum(res3[0] != -1))/RUNTIME/NUMP
                # resUtil4 = sum(sum(res4[0] != -1))/RUNTIME/NUMP
                # resUtil5 = sum(sum(res5[0] != -1))/RUNTIME/NUMP
                # resUtil6 = sum(sum(res6[0] != -1))/RUNTIME/NUMP
                # rawUtil = sum(taskSet[:, _C]/taskSet[:, _T])/NUMP
                
                # print(resUtil1, resUtil2, resUtil3, resUtil4, resUtil5, resUtil6, rawUtil)

                # # charger utilization
                # sum(sum(res1[1] != -1))/RUNTIME/NUMC
                # sum(taskSet[:, _CG]  / taskSet[:, _T])/NUMC

                if sum(res1[2] > taskSet[:, _RSW]):
                    diff = (res1[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res1[3] > taskSet[:, _RCG]):
                    diff = (res1[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)
                if sum(res2[2] > taskSet[:, _RSW]):
                    diff = (res2[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res2[3] > taskSet[:, _RCG]):
                    diff = (res2[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)
                if sum(res3[2] > taskSet[:, _RSW]):
                    diff = (res3[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res3[3] > taskSet[:, _RCG]):
                    diff = (res3[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)
                if sum(res4[2] > taskSet[:, _RSW]):
                    diff = (res4[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res4[3] > taskSet[:, _RCG]):
                    diff = (res4[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)
                if sum(res5[2] > taskSet[:, _RSW]):
                    diff = (res5[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res5[3] > taskSet[:, _RCG]):
                    diff = (res5[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)
                if sum(res6[2] > taskSet[:, _RSW]):
                    diff = (res6[2] - taskSet[:, _RSW])
                    print(diff[diff > 0])
                    print("FAIL1", params, chargerNUM)
                if sum(res6[3] > taskSet[:, _RCG]):
                    diff = (res6[3] - taskSet[:, _RCG])
                    print(diff[diff > 0])
                    print("FAIL2", params, chargerNUM)

                # if sum([res1[-1], res2[-1], res3[-1], res4[-1]]) >= 1:
                #     print(res1[-1], res2[-1], res3[-1], res4[-1])
                    
            else:

                analysisFail2 += -1

        else:

            analysisFail1 += -1

    print(1)

if __name__ == "__main__":

    # FAIL2 [0.1, 5, 3, 1000, 1, 100, 1, 1, 1, 5], 3

    # UTIL, NUMT, NUMP = 0.1, 3, 3
    # UTIL, NUMT, NUMP = 0.1, 5, 3
    # params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
    # numcLi = [1, 2, 3, 4]
    # for numc in numcLi:
    #     result = mainRunner(params, numc)


    utilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    numtLi = [3, 4, 5]
    numpLi = [1, 2, 3, 4]
    numcLi = [1, 2, 3, 4]

    # utilLi = [0.6, 0.7]
    # numtLi = [4, 5]
    # numpLi = [3, 4]
    # numcLi = [3, 4]
    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc)

    print("a")