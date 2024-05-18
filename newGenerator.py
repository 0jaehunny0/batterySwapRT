import sys
import numpy as np
from multiprocessing import Pool
import pickle
import os
import scipy.io
from itertools import repeat
import functools


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
# NUMB = 10   # bat   # [1- ] number of batteries --> need to be categorized later

# camelCase
n, util, nSets = None, None, None

# Directory
taskDir = "taskSet2/"
scenarioDir = "scenario2/"

def TCCD(seed_id, params):

    new_seed = seed_id #+ 77777 #777777 77777 19950717

    while True:

        np.random.seed(new_seed) # set random seed
        sUtil, cUtil, NUMT, NUMP, NUMC, NUMS = params

        CSW = np.random.randint(30, 100+1,  NUMT)    

        SW_UTIL = np.random.random(NUMT)

        SW_UTIL = SW_UTIL/sum(SW_UTIL) * sUtil * NUMP

        T = np.round(CSW/SW_UTIL)

        if sum(T < 1):
            print("stop")
            assert sum(T < 1)
            assert sum(T >= 1)


        CCG = np.ones(NUMT) * 600

        targetUtil = cUtil * NUMC

        lastIdx = -1
        while True:
            if sum(CCG/T) >= targetUtil:
                if lastIdx != -1:
                    if abs(targetUtil - sum(CCG/T)) < abs(targetUtil - lastUtil):
                        CCG[lastIdx] -= -1
                break
            idx = np.random.randint(0, NUMT)

            if CCG[idx] < 6001:
                lastUtil = sum(CCG/T)
                CCG[idx] += 1
                lastIdx = idx

            if sum(CCG) == 6001 * NUMT:
                break

        if sum(CCG) == 600 * NUMT or sum(CCG) == 6001 * NUMT:
            new_seed += 1000
            if new_seed >= 2**32:
                new_seed = (new_seed % 2**32)

            continue

        D = np.ones(NUMT) * 300

        ID = np.int32(np.arange(NUMT))

        print(seed_id, "end") 
        shapeT = np.array([T, CSW, D, ID,  CCG]).T # period, wcet, deadline, id, WCET cg
        return list(shapeT)

def nameCreator(parmas):
    return "_".join(map(str, parmas))

def pickleSaver(name, data): # save data
    with open(taskDir+name+".pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def rawPickleSaver(name, data): # save data
    with open(name+".pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def pickleLoader(name): # load data
    with open(taskDir+name+".pickle", 'rb') as f:
        data = pickle.load(f)
    return data

def rawPickleLoader(name): # load data
    with open(name+".pickle", 'rb') as f:
        data = pickle.load(f)
    return data

def taskSetsGenerator(params):
    assert len(params) == 6, "parameters: sUtil, cUtil, numt, nump, numc, NUMS, MINT, MAXT"
    global sUtil, cUtil, numt, nump, numc, NUMS
    sUtil, cUtil, numt, nump, numc, NUMS= params

    if not os.path.exists(taskDir):
        os.makedirs(taskDir)

    if not os.path.exists(scenarioDir):
        os.makedirs(scenarioDir)
    
    myPool = Pool(8) # multiprocessing
    result = np.array(myPool.map(functools.partial(TCCD, params=params), list(range(0,NUMS))))
    myPool.close()
    myPool.join()

    print(params, "end")

    name = nameCreator(params)
    pickleSaver(name, result)

    return result

if __name__ == "__main__":

    # stationUtilLi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # chargerUtilLi = [0.5]
    # numtLi = [4]
    # numpLi = [2]
    # numcLi = [30]
    # for sUtil in stationUtilLi:
    #     for cUtil in chargerUtilLi:
    #         for numt in numtLi:
    #             for nump in numpLi:
    #                 for numc in numcLi:
    #                     params = [sUtil, cUtil, numt, nump, numc, NUMS]
    #                     result = taskSetsGenerator(params)

    stationUtilLi = [0.5]
    chargerUtilLi = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    numtLi = [4]
    numpLi = [2]
    numcLi = [30]
    # for sUtil in stationUtilLi:
    #     for cUtil in chargerUtilLi:
    #         for numt in numtLi:
    #             for nump in numpLi:
    #                 for numc in numcLi:
    #                     params = [sUtil, cUtil, numt, nump, numc, NUMS]
    #                     result = taskSetsGenerator(params)


    # stationUtilLi = [0.5]
    # chargerUtilLi = [0.5]
    # numtLi = np.arange(1,11,1)
    # numpLi = [2]
    # numcLi = [30]
    # for sUtil in stationUtilLi:
    #     for cUtil in chargerUtilLi:
    #         for numt in numtLi:
    #             for nump in numpLi:
    #                 for numc in numcLi:
    #                     params = [sUtil, cUtil, numt, nump, numc, NUMS]
    #                     result = taskSetsGenerator(params)

    stationUtilLi = [0.5] #19950717
    chargerUtilLi = [0.5]
    numtLi = [4]
    numpLi = [1,2,3,4,5]
    numcLi = [30]
    for sUtil in stationUtilLi:
        for cUtil in chargerUtilLi:
            for numt in numtLi:
                for nump in numpLi:
                    for numc in numcLi:
                        params = [sUtil, cUtil, numt, nump, numc, NUMS]
                        result = taskSetsGenerator(params)

    # stationUtilLi = [0.5]
    # chargerUtilLi = [0.5]
    # numtLi = [4]
    # numpLi = [2]
    # numcLi = [10, 50]
    # for sUtil in stationUtilLi:
    #     for cUtil in chargerUtilLi:
    #         for numt in numtLi:
    #             for nump in numpLi:
    #                 for numc in numcLi:
    #                     params = [sUtil, cUtil, numt, nump, numc, NUMS]
    #                     result = taskSetsGenerator(params)

    print(1)

"""
[1] "Measuring the Performance of Schedulability Tests"

[2] "Priority Assignment for Global Fixed Priority Pre-Emptive Scheduling in Multiprocessor Real-Time Systems"

[3] "Techniques For The Synthesis Of Multiprocessor Tasksets"

[4] my idea (Jaeheon Kwak)

"""