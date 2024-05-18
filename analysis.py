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



def analysisCG(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = VD_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    NCG = numc

    while True:

        prevR = taskSet[:, _RCG]
        NBL = np.floor((np.fmax(0, prevR - T)) / T)
        BL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))

        newRList = []
        for idx in range(numt):

            first = C[idx]
            second = np.ceil ((NBL[idx] * C[idx] + BL[idx]) / NCG)
            
            fast = sum(C) + sum(NBL*C) + sum(BL) - C[idx] - NBL[idx] * C[idx] - BL[idx]
            thrid = np.ceil(fast / NCG)

            newR = np.ceil( first + second + thrid )
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:

            if sum(taskSet[:, _RCG] >= taskSet[:,_VD]) >= 1:
                return [-1]
            
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] >= taskSet[:,_VD]) >= 1:
                return [-1]
            
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList

def virtualDeadline(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # new room
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    for idx in range(numt):

        VD = batterySet[idx] * taskSet[idx, _T] - taskSet[idx, _RSW]

        while True:

            if batterySet[idx] >= np.ceil( (taskSet[idx, _RSW] + VD) / taskSet[idx, _T]):
                break

            VD -= 1
        
        taskSet[idx, _VD] = VD
    
    return taskSet

def analysisSW(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    NSW = nump

    while True:

        prevR = taskSet[:, _RSW]
        NBL = np.floor((np.fmax(0, prevR - T)) / T)
        BL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))

        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = np.ceil ((NBL[idx] * C[idx] + BL[idx]) / NSW)
            
            fast = sum(C) + sum(NBL*C) + sum(BL) - C[idx] - NBL[idx] * C[idx] - BL[idx]
            thrid = np.ceil(fast / NSW)

            newR = np.ceil( first + second + thrid )
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList