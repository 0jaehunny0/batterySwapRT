import numpy as np
import operator
import copy
from numba import jit, njit
# period, wcet, deadline, gumbel, power, release, seed, slack, fake wcet
_T = 0
_C = 1  #  C^SW
_D = 2
_ID = 3
_CG = 4 # C^CG
_RSW = 5#
_VD = 6
_RCG = 7
_CG = 8


def analysisCG(taskSet, params, batterySet, C_CG, chargerNUM):

    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD = params

    # init R_x(i) = VD_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    # add C_CG in matrix
    taskSet = np.hstack((taskSet, np.array([C_CG]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    NCG = chargerNUM

    while True:

        prevR = taskSet[:, _RCG]
        NBL = np.floor((np.fmax(0, prevR - T)) / T)
        BL = ((np.fmax(0, prevR - T)) % T)

        common = sum( (NBL * C) + BL + C )

        newRList = []
        for idx in range(NUMT):
            newR = np.ceil(common / NCG + (NCG - 1)/NCG * taskSet[idx, _CG])
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == NUMT:

            if sum(taskSet[:, _RCG] >= taskSet[:,_VD]) >= 1:
                return [-1]
            
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == NUMT:

            if sum(taskSet[:, _RCG] >= taskSet[:,_VD]) >= 1:
                return [-1]
            
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList

def virtualDeadline(taskSet, params, batterySet, C_CG, chargerNUM):

    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD = params

    # new room
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    for idx in range(NUMT):

        VD = batterySet[idx] * taskSet[idx, _T] - taskSet[idx, _RSW]

        while True:

            if batterySet[idx] >= np.ceil( (taskSet[idx, _RSW] + VD) / taskSet[idx, _T]):
                break

            VD -= 1
        
        taskSet[idx, _VD] = VD
    
    return taskSet

def analysisSW(taskSet, params, batterySet, C_CG, chargerNUM):

    UTIL, NUMT, NUMP, NUMS, MINT, MAXT, OPTS, OPTD, MIND, MAXD = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    NSW = NUMP

    while True:

        prevR = taskSet[:, _RSW]
        NBL = np.floor((np.fmax(0, prevR - T)) / T)
        BL = ((np.fmax(0, prevR - T)) % T)

        common = sum( (NBL * C) + BL + C )

        newRList = []
        for idx in range(NUMT):
            newR = np.ceil(common / NSW + (NSW - 1)/NSW * taskSet[idx, _C])
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == NUMT:
            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == NUMT:

            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList