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


RUNTIME = 10000
_T = 0
_C = 1
_D = 2
_G = 3
_P = 4
_R = 5
_E = 6
_S = 7
_F = 8
_M = 9
_ID = 10
_PF = 11
_PC = 12

# Basic parameters
UTIL = 0.75 # util  # (0-1] target utilization
NUMT = 3    # n     # [1- ] number of tasks / number of car types
NUMP = 4    # m     # [1- ] number of processors / number of swap stations
NUMS = 1000 # nSets # [1- ] number of sets / number of scenarios
MINT = 1    # minT  # [1- ] minimum periods 
MAXT = 100  # maxT  # [1- ] maximum periods

# Optional parameters
OPTS = 1    #      # [0- ] random seed value
OPTD = 0    #      # [0,1] 0: implicit-deadlines, 1: constrained-deadlines 

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

def mainRunner(params):
    assert len(params) == 8, "parameters: UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD"
    global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD
    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD = params

    name = nameCreator(params)

    myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, OPTG, OPTP])

    batterySet = [5,5,5]
    C_CG = [1,1,2]
    chargerNUM = 4

    RM = 0
    for i in range(0, NUMS):
        analysisResultRM = analysisRM(myRes[i, :, :], NUMP)

        RM += analysisResultRM

        if analysisResultRM == 0:
            res3 = FIFOrunnerLeakyQueue(myRes[i, :, :], NUMP, RUNTIME, batterySet, C_CG, chargerNUM)
            res1 = FIFOrunner(myRes[i, :, :], NUMP, RUNTIME, batterySet, C_CG, chargerNUM)
            res2 = FIFOrunner2(myRes[i, :, :], NUMP, RUNTIME, batterySet, C_CG, chargerNUM)


            """ checking utiization """
            taskSet = myRes[i, :, :]
            resUtil = sum(sum(res1[0] != -1))/RUNTIME
            sum(taskSet[:, _C]/taskSet[:, _T])/3




# def newRunnerCompare(util, numt, nump, optp): # RM EDF 동시 통과하는 taskset generation
#     global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, OPTG, OPTP
#     UTIL, NUMT, NUMP, OPTP = util, numt, nump, optp
#     params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, OPTG, OPTP]
#     name = nameCreator(params)

#     myRes = loadByName(name) # taskSetLoader([UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, OPTG, OPTP])

#     RM = 0
#     resultCount = 0
#     methodLi = []
#     maxPLi = []
#     idleLi = []
#     myCnt = 0
    
#     for i in range(0, NUMS):
        
#         analysisTestRM = analysisRM(myRes[i, :, :, :], NUMP)
#         analysisTestEDF = analysisEDF(myRes[i, :, :, :], NUMP)

#         if analysisTestRM == 0 & analysisTestEDF == 0:
#             analysisTest = 0
#         else:
#             analysisTest = -1

#         RM += analysisTest
        
#         if analysisTest == 0:

#             EDFFLAG = 1
#             RMFLAG = 2

#             # NWCFmadeEDF = makeNWCF(myRes[i, :, :, :], NUMP, _PF, _PF, -1 , -1)
#             # NWCFmadeRM = makeNWCF_RM(myRes[i, :, :, :], NUMP, _PF, _PF, -1 , -1)

#             # # newRSTEDF, newRSTEDFutil = dateNewSlackTemp(NWCFmadeEDF, NUMP, RUNTIME, OPTS, 0, _PC, -1, EDFFLAG)
#             # # newRSTRM, newRSTRMutil = dateNewSlackTemp(NWCFmadeRM, NUMP, RUNTIME, OPTS, 0, _PC, -1, RMFLAG)

#             # # # NWCFmadeEDF = makeNWCF(myRes[i, :, :, :], NUMP, _P, _F, -1 , 1)
#             # # # NWCFmadeRM = makeNWCF_RM(myRes[i, :, :, :], NUMP, _P, _F, -1 , 1)

#             # # # newRSTEDF2, newRSTEDFutil = dateNewSlackTemp2(NWCFmadeEDF, NUMP, RUNTIME, OPTS, 0, _P, -1, EDFFLAG)
#             # # # newRSTRM2, newRSTRMutil = dateNewSlackTemp2(NWCFmadeRM, NUMP, RUNTIME, OPTS, 0, _P, -1, RMFLAG)

#             # newRSTEDF2, newRSTEDFutil = dateNewSlackTemp2(NWCFmadeEDF, NUMP, RUNTIME, OPTS, 0, _PC, -1, EDFFLAG)
#             # newRSTRM2, newRSTRMutil = dateNewSlackTemp2(NWCFmadeRM, NUMP, RUNTIME, OPTS, 0, _PC, -1, RMFLAG)

#             # resultLi = [newRSTEDF2, newRSTRM2]

#             # # print(newRSTEDF.var(), newRSTEDF2.var(), newRSTRM.var(), newRSTRM2.var())

#             NWCFmadeEDF = makeNWCF(myRes[i, :, :, :], NUMP, _P, _F, -1 , 1)
#             NWCFmadeRM = makeNWCF_RM(myRes[i, :, :, :], NUMP, _P, _F, -1 , 1)

#             vanillaEDF, vanillaEDFutil = dateVanillaRET(myRes[i, :, :, :], NUMP, RUNTIME, OPTS, 2, _P, -1, EDFFLAG)
#             vanillaRM, vanillaRMutil = dateVanillaRET(myRes[i, :, :, :], NUMP, RUNTIME, OPTS, 2, _P, -1, RMFLAG)

#             vanillaRETEDF, vanillaRETEDFutil = dateVanillaRET(NWCFmadeEDF, NUMP, RUNTIME, OPTS, 0, _P, -1, EDFFLAG)
#             vanillaRETRM, vanillaRETRMutil = dateVanillaRET(NWCFmadeRM, NUMP, RUNTIME, OPTS, 0, _P, -1, RMFLAG)

#             NWCFmadeEDF = makeNWCF(myRes[i, :, :, :], NUMP, _PC, _PF, -1 , 1)
#             NWCFmadeRM = makeNWCF_RM(myRes[i, :, :, :], NUMP, _PC, _PF, -1 , 1)

#             newRSTwithoutRETEDF, newRSTwithoutRETEDFutil = dateNewSlack(myRes[i, :, :, :], NUMP, RUNTIME, OPTS, 0, _PC, -1, EDFFLAG)
#             newRSTwithoutRETRM, newRSTwithoutRETRMutil = dateNewSlack(myRes[i, :, :, :], NUMP, RUNTIME, OPTS, 0, _PC, -1, RMFLAG)

#             newRSTEDF, newRSTEDFutil = dateNewSlack(NWCFmadeEDF, NUMP, RUNTIME, OPTS, 0, _PC, -1, EDFFLAG)
#             newRSTRM, newRSTRMutil = dateNewSlack(NWCFmadeRM, NUMP, RUNTIME, OPTS, 0, _PC, -1, RMFLAG)

#             if myCnt < 10:
#                 matSaver("dateCompare_"+name+"_"+str(i), dict(util = util, vanillaEDF = vanillaEDF, vanillaRETEDF = vanillaRETEDF, newRSTwithoutRETEDF = newRSTwithoutRETEDF, newRSTEDF = newRSTEDF, vanillaRM = vanillaRM, vanillaRETRM = vanillaRETRM, newRSTwithoutRETRM = newRSTwithoutRETRM, newRSTRM = newRSTRM))

#             resultLi = [vanillaEDF, vanillaRETEDF, newRSTwithoutRETEDF, newRSTEDF, vanillaRM, vanillaRETRM, newRSTwithoutRETRM, newRSTRM]

#             # saves xxx utilization
#             resultLi2 = [vanillaEDFutil, vanillaRETEDFutil, newRSTwithoutRETEDFutil, newRSTEDFutil, vanillaRMutil, vanillaRETRMutil, newRSTwithoutRETRMutil, newRSTRMutil]

#             resultLi = np.array(resultLi, dtype=np.float64)
#             resultLi = resultLi / resultLi.mean(axis=1)[:, None]
#             resultLi *= OPTP / (0.8 / UTIL) * 4
#             method = list(np.array(resultLi, dtype=np.float64).var(axis=1))
#             maxP = list(np.array(resultLi).max(axis=1))

#             resultLi2 = np.array(resultLi2, dtype=np.float64)
#             half = int(RUNTIME/2)
#             (resultLi2[:,:,:half] == -1).sum(axis=2) / half
#             idle = list(((resultLi2[:,:,:half] == -1).sum(axis=2) / half).mean(axis=1))

#             methodLi.append(method)
#             maxPLi.append(maxP)
#             idleLi.append(idle)
        
#             print(method)
#             print(i, util, numt, nump)
#             resultCount += 1
#             myCnt += 1

#         if resultCount == 100:
#             break
        
#     print(RM, i)
#     method = list(np.array(methodLi).mean(axis=0))
#     maxP = list(np.array(maxPLi).mean(axis=0))
#     idle = list(np.array(idleLi).mean(axis=0))
#     methodLi.append(method)
#     maxPLi.append(maxP)
#     idleLi.append(idle)

#     version = ""
#     pickleSaver("dateCompare_"+name+"_methodLi"+str(version), np.array(methodLi))
#     pickleSaver("dateCompare_"+name+"_maxPLi"+str(version), np.array(maxPLi))
#     pickleSaver("dateCompare_"+name+"_idleLi"+str(version), np.array(idleLi))




# motivationDraw()
if __name__ == "__main__":

    utilLi = [0.2, 0.3, 0.4]
    numtLi = [3]
    numpLi = [3]
    
    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                UTIL, NUMT, NUMP = util, numt, nump
                params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD]
                result = mainRunner(params)