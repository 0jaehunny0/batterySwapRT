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
NUMB = 10   # bat   # [1- ] number of batteries --> need to be categorized later

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
    
    RM, RM2 = 0, 0
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

                res1 = FIFOrunner(taskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM)                

                resUtil = sum(sum(res1[0] != -1))/RUNTIME

                sum(taskSet[:, _C]/taskSet[:, _T])/NUMP


                if sum(res1[2] > taskSet[:, _RSW]):
                    print("FAIL")
                if sum(res1[3] > taskSet[:, _RCG]):
                    print("FAIL")

            else:

                RM2 += -1

        else:

            RM += -1

    print(1)

if __name__ == "__main__":


    # UTIL, NUMT, NUMP = 0.1, 3, 1
    # params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
    # result = mainRunner(params, 1)


    utilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    numtLi = [3, 4, 5]
    numpLi = [1, 2, 3, 4]
    numcLi = [1, 2, 3, 4]
    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                for numc in numcLi:
                    UTIL, NUMT, NUMP = util, numt, nump
                    params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD]
                    result = mainRunner(params, numc)



    print("a")