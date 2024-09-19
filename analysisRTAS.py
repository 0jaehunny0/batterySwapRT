import numpy as np
import operator
import copy
from numba import jit, njit
from heapq import heappushpop, heapify
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
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                print(taskSet[:, _RSW])
                return [-1]
            print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                print(taskSet[:, _RSW])
                return [-1]
            print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList


def NEWanalysisSWasd(taskSet, params, batterySet):

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
        if C[idx] > T[idx]:
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
        if D[idx] > T[idx]:
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
            if C[idx] > T[idx]:
                for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                    onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
            if prevRy[idx] > T[idx]:
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
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
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

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NSW, dtype=np.int32)) 
            heapify(BF)            
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                val = C[idx] - a * T[idx]
                if val > C[idx]:
                    oneBL += C[idx]
                elif val > BF[0]:
                    heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                    val = prevRy[idx2] - a * T[idx2]
                    if val > C[idx2]:
                        oneBL += C[idx2]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RSW] = newRList
            prevR = taskSet[:, _RSW]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList
        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)

def RTASanalysisSW2(taskSet, params, batterySet):
    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump    

    # response time = C
    taskSet[:,_RSW] = taskSet[:,_C]

    prevR = C.copy() # slack = D - C / S = D - R / R = D - S / saves last resopnse time

    update = True
    while update:
        update = False

        for idx in range(numt): # for k = 0 to n do
            while True: 
                oneBL = 0
                if prevR[idx] > T[idx]: # cacluate BL*x (L)
                    for a in range(1, int(np.floor(prevR[idx]/T[idx])) + 1):
                        val = prevR[idx] - a * T[idx]
                        oneBL += min(C[idx], val)
                for idx2 in range(numt): # cacluate BL*y (Ry) except for x
                    if idx == idx2: continue 
                    if prevR[idx2] > T[idx2]:
                        for a in range(1, int(np.floor(prevR[idx2]/T[idx2])) + 1):
                            val = prevR[idx2] - a * T[idx2]
                            oneBL += min(C[idx2], val)                    
                L = np.floor(C[idx] + (oneBL + sum(C) - C[idx]) / NSW) # equation 3
                if L == prevR[idx]: 
                    prevR[idx] = L
                    break
                prevR[idx] = L
            
            if L > D[idx]:
                return [-1] # unschedulable
            if L > taskSet[idx, _RSW]:
                taskSet[idx, _RSW] = L
                update = True
    return taskSet

def RTASanalysisSW3(taskSet, params, batterySet):
    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump    

    # response time = C
    taskSet[:,_RSW] = taskSet[:,_C]

    prevRx = C.copy() # slack = D - C / S = D - R / R = D - S / saves last resopnse time of x
    prevRy = D.copy() # slack = D - C / S = D - R / R = D - S / saves last resopnse time of y

    update = True
    while update:
        update = False

        for idx in range(numt): # for k = 0 to n do
            while True: 
                oneBL = 0
                if prevRx[idx] > T[idx]: # cacluate BL*x (L)
                    for a in range(1, int(np.floor(prevRx[idx]/T[idx])) + 1):
                        val = prevRx[idx] - a * T[idx]
                        oneBL += min(C[idx], val)
                for idx2 in range(numt): # cacluate BL*y (Ry) except for x
                    if idx == idx2: continue 
                    if prevRy[idx2] > T[idx2]:
                        for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                            val = prevRy[idx2] - a * T[idx2]
                            oneBL += min(C[idx2], val)                    
                L = np.floor(C[idx] + (oneBL + sum(C) - C[idx]) / NSW) # equation 3
                if L == prevRx[idx]: 
                    prevRx[idx] = L
                    break
                prevRx[idx] = L
            
            if L > D[idx]:
                return [-1] # unschedulable
            if L > taskSet[idx, _RSW]:
                taskSet[idx, _RSW] = L
                update = True
        prevRy = prevRx.copy()
    return taskSet


def RTASanalysisSW(taskSet, params, batterySet):
    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump    

    # response time = C
    taskSet[:,_RSW] = taskSet[:,_C]

    prevR = C * 0
    prevRy = C.copy() # slack = D - C / S = D - R / R = D - S 

    update = True
    firstTry = True
    while update:
        update = False
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NSW, dtype=np.int32)) 
            heapify(BF)        
            if  prevRy[idx] > T[idx]:
                for a in range(1, int(np.floor(prevRy[idx]/T[idx])) + 1):
                    val = prevRy[idx] - a * T[idx]
                    if val > C[idx]:
                        oneBL += C[idx]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                if prevRy[idx2] > T[idx2]:
                    for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                        val = prevRy[idx2] - a * T[idx2]
                        if val > C[idx2]:
                            oneBL += C[idx2]
                        elif val > BF[0]:
                            heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
            if newR > D[idx]:
                return [-1]

            if newR > taskSet[idx, _RSW]:
                taskSet[idx, _RSW] = newR
                prevRy[idx] = newR
                update = True
        newRList = np.array(newRList)

        # if firstTry:
        #     firstTry = False
        #     taskSet[:, _RSW] = newRList
        #     prevR = taskSet[:, _RSW]
        #     prevRy = np.fmin(D, prevR)
        #     update = True
        #     continue


        prevR = taskSet[:, _RSW].copy()

    return taskSet

def NEWanalysisSW2(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NSW, dtype=np.int32)) 
            heapify(BF)            
            if  prevR[idx] > T[idx]:
                for a in range(1, int(np.floor(prevR[idx]/T[idx])) + 1):
                    val = prevR[idx] - a * T[idx]
                    if val > C[idx]:
                        oneBL += C[idx]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                if prevRy[idx2] > T[idx2]:
                    for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                        val = prevRy[idx2] - a * T[idx2]
                        if val > C[idx2]:
                            oneBL += C[idx2]
                        elif val > BF[0]:
                            heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RSW] = newRList
            prevR = taskSet[:, _RSW]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        # if sum(prevR <= newRList) == numt:

        #     if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
        #         # print(taskSet[:, _RSW])
        #         return [-1]
        #     # print(taskSet[:, _RSW])
        #     return taskSet # schedulable, conversed

        #     # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList
        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)

def NEWanalysisSW4(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NSW*10, dtype=np.int32)) 
            heapify(BF)            
            if  prevR[idx] > T[idx]:
                for a in range(1, int(np.floor(prevR[idx]/T[idx])) + 1):
                    val = prevR[idx] - a * T[idx]
                    if val > C[idx]:
                        oneBL += C[idx]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                if prevRy[idx2] > T[idx2]:
                    for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                        val = prevRy[idx2] - a * T[idx2]
                        if val > C[idx2]:
                            oneBL += C[idx2]
                        elif val > BF[0]:
                            heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RSW] = newRList
            prevR = taskSet[:, _RSW]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        # if sum(prevR <= newRList) == numt:

        #     if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
        #         # print(taskSet[:, _RSW])
        #         return [-1]
        #     # print(taskSet[:, _RSW])
        #     return taskSet # schedulable, conversed

        #     # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList
        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)


def NEWanalysisCG0(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    D = taskSet[:,_VD]
    NCG = numc

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        BLy = []
        for idx in range(numt):
            oneBL = 0    
            oneBLy = 0                
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                oneBL += C[idx]
            for a in range(1, int(np.floor(prevRy[idx]/T[idx])) + 1):
                oneBLy += C[idx]
            BL.append(oneBL)
            BLy.append(oneBLy)
        BL = np.array(BL)
        BLy = np.array(BLy)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) + sum(BLy) - C[idx] - BLy[idx]
            newR = np.floor( first + (second + thrid)/NCG)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RCG] = newRList
            prevR = taskSet[:, _RCG]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                # print(taskSet[:, _RCG])
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                # print(taskSet[:, _RCG])
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList
        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)

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
        if C[idx] > T[idx]:
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
        if D[idx] > T[idx]:
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
            if C[idx] > T[idx]:
                for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                    onemyBL += np.min([C[idx], C[idx] - a * T[idx]])
            if prevRy[idx] > T[idx]:
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
            newR = np.floor( first + (second + thrid)/NCG)
            newRList.append(newR)
        newRList = np.array(newRList)

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                return [-1]
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList



def NEWanalysisCG2(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    D = taskSet[:,_VD]
    NCG = numc

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NCG, dtype=np.int32)) 
            heapify(BF)            
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                val = C[idx] - a * T[idx]
                if val > C[idx]:
                    oneBL += C[idx]
                elif val > BF[0]:
                    heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                    val = prevRy[idx2] - a * T[idx2]
                    if val > C[idx2]:
                        oneBL += C[idx2]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NCG)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RCG] = newRList
            prevR = taskSet[:, _RCG]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                # print(taskSet[:, _RCG])
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
                # print(taskSet[:, _RCG])
                return [-1]
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList
        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)


def NEWanalysisSW0(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _D]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        BLy = []
        for idx in range(numt):
            oneBL = 0    
            oneBLy = 0                
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                oneBL += C[idx]
            for a in range(1, int(np.floor(prevRy[idx]/T[idx])) + 1):
                oneBLy += C[idx]
            BL.append(oneBL)
            BLy.append(oneBLy)
        BL = np.array(BL)
        BLy = np.array(BLy)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) + sum(BLy) - C[idx] - BLy[idx]
            newR = np.floor( first + (second + thrid)/NSW)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RSW] = newRList
            prevR = taskSet[:, _RSW]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                return [-1]
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList
        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)




def NEWanalysisCG3(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    taskSet = np.hstack((taskSet, np.array([taskSet[:, _VD]]).T))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    D = taskSet[:,_VD]
    NCG = numc

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        BL = []
        for idx in range(numt):
            oneBL = 0            
            BF = list(np.zeros(NCG, dtype=np.int32)) 
            heapify(BF)            
            for a in range(1, int(np.floor(C[idx]/T[idx])) + 1):
                val = C[idx] - a * T[idx]
                if val > C[idx]:
                    oneBL += C[idx]
                elif val > BF[0]:
                    heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                for a in range(1, int(np.floor(prevRy[idx2]/T[idx2])) + 1):
                    val = prevRy[idx2] - a * T[idx2]
                    if val > C[idx2]:
                        oneBL += C[idx2]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            oneBL += sum(BF)
            BL.append(oneBL)
        BL = np.array(BL)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.floor( first + (second + thrid)/NCG)
            newRList.append(newR)
        newRList = np.array(newRList)

        if firstTry:
            firstTry = False
            taskSet[:, _RCG] = newRList
            prevR = taskSet[:, _RCG]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            # if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
            #     # print(taskSet[:, _RCG])
            #     return [-1]
            # # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            # if sum(taskSet[:, _RCG] > taskSet[:,_VD]) >= 1:
            #     # print(taskSet[:, _RCG])
            #     return [-1]
            # # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList
        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)


# @njit(fastmath=True)
def NEWanalysisSW3(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    
    ZZ = [list(taskSet[:, _D])]
    YY = np.array(ZZ, dtype=np.int32)
    XX = YY.T
    taskSet = np.hstack((taskSet, XX))

    T = taskSet[:,_T]
    C = taskSet[:,_C]
    D = taskSet[:,_D]
    NSW = nump

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        # print(1)
        BL = []
        for idx in range(numt):
            oneBL = 0 
            BFX  = np.zeros(int(NSW), dtype=np.int32)        
            BF = list(BFX) 
            heapify(BF)            
            for a in range(1, np.int32(np.floor(C[idx]/T[idx])) + 1):
                val = np.int32(C[idx] - a * T[idx])
                if val > C[idx]:
                    oneBL += C[idx]
                elif val > BF[0]:
                    heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                for a in range(1, np.int32(np.floor(prevRy[idx2]/T[idx2])) + 1):
                    val = np.int32(prevRy[idx2] - a * T[idx2])
                    if val > C[idx2]:
                        oneBL += C[idx2]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            oneBL += sum(BF)
            oneBL = np.int32(oneBL)
            BL.append(oneBL)
        BL = np.array(BL, dtype=np.int32)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.int32(np.floor( first + (second + thrid)/NSW))
            newRList.append(newR)
        newRList = np.array(newRList, dtype=np.int32)

        if firstTry:
            firstTry = False
            taskSet[:, _RSW] = newRList
            prevR = taskSet[:, _RSW]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                taskSet[0][0] = 0
                return taskSet
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:

            if sum(taskSet[:, _RSW] > taskSet[:,_D]) >= 1:
                # print(taskSet[:, _RSW])
                taskSet[0][0] = 0
                return taskSet
            # print(taskSet[:, _RSW])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RSW] = newRList
        prevR = taskSet[:, _RSW]
        prevRy = np.fmin(D, prevR)



@njit(fastmath=True)
def NEWanalysisCG4(taskSet, params, batterySet):

    sUtil, cUtil, numt, nump, numc, NUMS = params

    # init R_x(i) = D_x
    
    ZZ = [list(taskSet[:, _D])]
    YY = np.array(ZZ, dtype=np.int32)
    XX = YY.T
    taskSet = np.hstack((taskSet, XX))

    T = taskSet[:,_T]
    C = taskSet[:,_CG]
    D = taskSet[:,_VD]
    NCG = numc

    prevR = C
    prevRy = D
    firstTry = True

    while True:
        # print(1)
        BL = []
        for idx in range(numt):
            oneBL = 0 
            BFX  = np.zeros(int(NCG), dtype=np.int32)        
            BF = list(BFX) 
            heapify(BF)            
            for a in range(1, np.int32(np.floor(C[idx]/T[idx])) + 1):
                val = np.int32(C[idx] - a * T[idx])
                if val > C[idx]:
                    oneBL += C[idx]
                elif val > BF[0]:
                    heappushpop(BF, val)
            for idx2 in range(numt):
                if idx == idx2: continue
                for a in range(1, np.int32(np.floor(prevRy[idx2]/T[idx2])) + 1):
                    val = np.int32(prevRy[idx2] - a * T[idx2])
                    if val > C[idx2]:
                        oneBL += C[idx2]
                    elif val > BF[0]:
                        heappushpop(BF, val)
            oneBL += sum(BF)
            oneBL = np.int32(oneBL)
            BL.append(oneBL)
        BL = np.array(BL, dtype=np.int32)
        newRList = []
        for idx in range(numt):
            first = C[idx]
            second = BL[idx]
            thrid = sum(C) - C[idx]
            newR = np.int32(np.floor( first + (second + thrid)/NCG))
            newRList.append(newR)
        newRList = np.array(newRList, dtype=np.int32)

        if firstTry:
            firstTry = False
            taskSet[:, _RCG] = newRList
            prevR = taskSet[:, _RCG]
            prevRy = np.fmin(D, prevR)
            continue

        if sum(prevR == newRList) == numt:
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

        if sum(prevR <= newRList) == numt:
            # print(taskSet[:, _RCG])
            return taskSet # schedulable, conversed

            # return [-1] # unsched
        
        taskSet[:, _RCG] = newRList
        prevR = taskSet[:, _RCG]
        prevRy = np.fmin(D, prevR)
