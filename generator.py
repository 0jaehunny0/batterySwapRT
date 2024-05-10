import sys
import numpy as np
from multiprocessing import Pool
import pickle
import os
import scipy.io
from itertools import repeat
import functools


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

# Directory
taskDir = "taskSet/"
scenarioDir = "scenario/"

def UUnifastDiscard(n, util, MAXT): # utilization generator 
    while True:
        oneSet = np.array([])
        sumU = util
        cnt = 0
        for i in range(1,n): # UUnifast / [1]
            if 1 / sumU > MAXT:
                oneSet = np.append(oneSet, sumU)
                break
            nextSumU = sumU * np.power( np.random.rand(), (1.0 / (n - i)) )
            while 1 / (sumU - nextSumU) > MAXT: # guarantee C >= 1 / [4]
                nextSumU = sumU * np.power( np.random.rand(), (1.0 / (n - i)) ) 
                cnt += 1
                if cnt > 10000:
                    break
            oneSet = np.append(oneSet, sumU - nextSumU)
            if sumU - nextSumU > 1:
                break
            sumU = nextSumU
        oneSet = np.append(oneSet, sumU)

        if oneSet.max() < 1 and (1 / oneSet).max() < MAXT and oneSet.size == n: # Discard / [2]
            return oneSet
        


def TC(n, util, oneUtil, MINT, MAXT): # T, C generator
    _low = np.max( ([MINT] * n, np.ceil(1 / oneUtil)), axis=0 ) # guarantee C >= 1 / [4]
    
    T = np.array([]) 
    for i in range(n): # log-uniform periods / [3]
        tempT = np.exp(np.random.uniform(low=np.log(MINT) , high=np.log(MAXT+0.49))) 
        while tempT < _low[i]:
            tempT = np.exp(np.random.uniform(low=np.log(MINT) , high=np.log(MAXT+0.49)))
        tempT = np.round(tempT)
        T = np.append(T, tempT)
    
    T = np.int32(T)
    C = np.int32(np.round(T*oneUtil))
    
    while (C/T).sum() < util: # guarantee utilization > UTIL * NUMP with less error
        idx = np.argsort( oneUtil - (C/T) )[::-1]
        for i in range(len(idx)):
            if C[i] + 1 < T[i]:
                C[i] = C[i] + 1
                break
        if i == len(idx) - 1:
            break
            # assert i != len(idx) - 1, "bad luck"

    return T, C

def TCDGP(seed_id, params): # T, C, D, G, P generator
    global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD
    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD = params
    n, util = NUMT, UTIL*NUMP
    cond = 2 * 1 / MAXT * n # guarantee C >= 1 / [4]
    assert cond < util, "hard to guarantee C >= 1: too low UTIL, MAXT and too high NUMT, MINT"

    np.random.seed(seed_id) # set random seed
    
    oneUtil = UUnifastDiscard(n, util, MAXT)
    
    T, C = TC(n, util, oneUtil, MINT, MAXT)
    
    if OPTD == 0: # implicit-deadlines
        D = T
    else: # constrained-deadlines
        D = np.int32(np.random.randint(low=C, high=T+1, size=n)) # uniform deadlines / [2]

    ID = np.int32(np.arange(n))

    print(seed_id, "end") 
    shapeT = np.array([T, C, D, ID]).T # period, wcet, deadline, id
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
    assert len(params) == 8, "parameters: UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD"
    global UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD
    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD = params

    if not os.path.exists(taskDir):
        os.makedirs(taskDir)

    if not os.path.exists(scenarioDir):
        os.makedirs(scenarioDir)
    
    myPool = Pool(8) # multiprocessing
    result = np.array(myPool.map(functools.partial(TCDGP, params=params), list(range(0,NUMS))))
    myPool.close()
    myPool.join()

    name = nameCreator(params)
    pickleSaver(name, result)

    return result

if __name__ == "__main__":

    utilLi = [0.2, 0.3, 0.4]
    numtLi = [3]
    numpLi = [3]

    for util in utilLi:
        for numt in numtLi:
            for nump in numpLi:
                UTIL, NUMT, NUMP = util, numt, nump
                params = [UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD]
                result = taskSetsGenerator(params)

    print(1)

"""
[1] "Measuring the Performance of Schedulability Tests"

[2] "Priority Assignment for Global Fixed Priority Pre-Emptive Scheduling in Multiprocessor Real-Time Systems"

[3] "Techniques For The Synthesis Of Multiprocessor Tasksets"

[4] my idea (Jaeheon Kwak)

"""