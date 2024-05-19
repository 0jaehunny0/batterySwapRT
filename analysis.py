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
    D = taskSet[:,_VD]
    NCG = numc

    """ first round Rx <- Cx, Ry <- Dy """
    prevR = taskSet[:, _RCG]
    myNBL = np.floor((np.fmax(0, C - T)) / T)
    myBL = np.fmin(C, ((np.fmax(0, C - T)) % T))
    NBL = np.floor((np.fmax(0, prevR - T)) / T)
    BL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))
    newRList = []
    for idx in range(numt):
        first = C[idx]
        second = np.ceil ((myNBL[idx] * C[idx] + myBL[idx]) / NCG)        
        
        fast = sum(C) + sum(NBL*C) + sum(BL) - C[idx] - NBL[idx] * C[idx] - BL[idx]
        thrid = np.ceil(fast / NCG)

        newR = np.ceil( first + second + thrid )
        newRList.append(newR)
    newRList = np.array(newRList)
    taskSet[:, _RCG] = newRList
    
    """ Loop"""
    while True:

        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)
        myNBL = np.floor((np.fmax(0, prevR - T)) / T)
        myBL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))
        NBL = np.floor((np.fmax(0, prevRy - T)) / T)
        BL = np.fmin(C, ((np.fmax(0, prevRy - T)) % T))

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

            if sum(taskSet[:, _RCG] + taskSet[:, _RSW] >= taskSet[:,_VD]) >= 1:
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] + taskSet[:, _RSW] >= taskSet[:,_VD]) >= 1:
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
    D = taskSet[:,_D]
    NSW = nump

    """ first round Rx <- Cx, Ry <- Dy """
    prevR = taskSet[:, _RSW]
    myNBL = np.floor((np.fmax(0, C - T)) / T)
    myBL = np.fmin(C, ((np.fmax(0, C - T)) % T))
    NBL = np.floor((np.fmax(0, prevR - T)) / T)
    BL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))
    newRList = []
    for idx in range(numt):
        first = C[idx]
        second = np.ceil ((myNBL[idx] * C[idx] + myBL[idx]) / NSW)        
        
        fast = sum(C) + sum(NBL*C) + sum(BL) - C[idx] - NBL[idx] * C[idx] - BL[idx]
        thrid = np.ceil(fast / NSW)

        newR = np.ceil( first + second + thrid )
        newRList.append(newR)
    newRList = np.array(newRList)
    taskSet[:, _RSW] = newRList
    
    """ Loop"""
    while True:

        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)
        myNBL = np.floor((np.fmax(0, prevR - T)) / T)
        myBL = np.fmin(C, ((np.fmax(0, prevR - T)) % T))
        NBL = np.floor((np.fmax(0, prevRy - T)) / T)
        BL = np.fmin(C, ((np.fmax(0, prevRy - T)) % T))

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
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList


def NEWanalysisSW(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump

    """ first round Rx <- Cx, Ry <- Dy """
    prevR = taskSet[:, _RSW]
    myBL = []
    BL = []
    for idx in range(numt):
        onemyBL = 0
        oneBL = 0
        if C[idx] <= T[idx]:
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
        if D[idx] <= T[idx]:
            for a in range(1, int(np.floor(D[idx]/T[idx])) + 1):
                oneBL += np.min([C[idx], D[idx] - a * T[idx]])
        myBL.append(onemyBL)
        BL.append(oneBL)
    myBL = np.array(myBL)
    BL = np.array(BL)
    newRList = []
    for idx in range(numt):
        first = C[idx]
        second = myBL[idx]
        thrid = sum(C) + sum(BL) - C[idx] - BL[idx]
        newR = np.ceil( first + (second + thrid)/NSW)
        newRList.append(newR)
    newRList = np.array(newRList)
    taskSet[:, _RSW] = newRList
    
    """ Loop"""
    while True:

        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)
        myBL = []
        BL = []
        for idx in range(numt):
            onemyBL = 0
            oneBL = 0
            if prevR[idx] <= T[idx]:
                for a in range(1, int(np.floor(prevR[idx]/T[idx])) + 1):
                    onemyBL += np.min([C[idx], prevR[idx] - a * T[idx]])
            if prevRy[idx] <= T[idx]:
                for a in range(1, int(np.floor(prevRy[idx]/T[idx])) + 1):
                    oneBL += np.min([C[idx], prevRy[idx] - a * T[idx]])
            myBL.append(onemyBL)
            BL.append(oneBL)
        myBL = np.array(myBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = myBL[idx]
            thrid = sum(C) + sum(BL) - C[idx] - BL[idx]
            newR = np.ceil( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] >= taskSet[:,_D]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList

def NEWanalysisCG(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    D = taskSet[:,_VD]
    NCG = numc

    """ first round Rx <- Cx, Ry <- Dy """
    prevR = taskSet[:, _RCG]
    myBL = []
    BL = []
    for idx in range(numt):
        onemyBL = 0
        oneBL = 0
        if C[idx] <= T[idx]:
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
        if D[idx] <= T[idx]:
            for a in range(1, int(np.floor(D[idx]/T[idx])) + 1):
                oneBL += np.min([C[idx], D[idx] - a * T[idx]])
        myBL.append(onemyBL)
        BL.append(oneBL)
    myBL = np.array(myBL)
    BL = np.array(BL)
    newRList = []
    for idx in range(numt):
        first = C[idx]
        second = myBL[idx]
        thrid = sum(C) + sum(BL) - C[idx] - BL[idx]
        newR = np.ceil( first + (second + thrid)/NCG)
        newRList.append(newR)
    newRList = np.array(newRList)
    taskSet[:, _RCG] = newRList
    
    """ Loop"""
    while True:

        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)
        myBL = []
        BL = []
        for idx in range(numt):
            onemyBL = 0
            oneBL = 0
            if prevR[idx] <= T[idx]:
                for a in range(1, int(np.floor(prevR[idx]/T[idx])) + 1):
                    onemyBL += np.min([C[idx], prevR[idx] - a * T[idx]])
            if prevRy[idx] <= T[idx]:
                for a in range(1, int(np.floor(prevRy[idx]/T[idx])) + 1):
                    oneBL += np.min([C[idx], prevRy[idx] - a * T[idx]])
            myBL.append(onemyBL)
            BL.append(oneBL)
        myBL = np.array(myBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = myBL[idx]
            thrid = sum(C) + sum(BL) - C[idx] - BL[idx]
            newR = np.ceil( first + (second + thrid)/NCG)
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RCG] + taskSet[:, _RSW] >= taskSet[:,_VD]) >= 1:
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] + taskSet[:, _RSW] >= taskSet[:,_VD]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList