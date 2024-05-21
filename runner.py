from asyncio import tasks
from numba import jit, njit, vectorize
import numpy as np
import copy

# from sklearn.utils import shuffle

_T = 0
_C = 1  #  C^SW
_D = 2
_ID = 3
_CG = 4 # C^CG
_RSW = 5#
_VD = 6
_RCG = 7
_CG = 4


@njit(fastmath=True)
def FIFOrunnerAHP1(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period, dynamic):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999999

    stationQ = np.empty((0, 6), dtype = np.int32)

    highQ = np.empty((0, 8), dtype = np.int32)
    lowQ = np.empty((0, 8), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 6), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 10), dtype = np.int32) * -1

    # measurement
    R_SW_list = np.empty((0), dtype = np.int64) #measured
    R_CG_list = np.empty((0), dtype = np.int64) #measured
    realR_SW_list = np.empty((0), dtype = np.int64) #measured
    realR_CG_list = np.empty((0), dtype = np.int64) #measured
    RSW_list = np.empty((0), dtype = np.int64) #guaranteed
    RCG_list = np.empty((0), dtype = np.int64) #guaranteed
    Preemption = 0
    totalRelease = 0
    totalhighCnt = 0
    acceptCnt = 0

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    TT = taskSet[idx, _T]
                    if time - prevRelease[idx] >= TT:
                        a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        acceptCnt += 1
                        stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter 
                    nextRelease[idx] += (TT + sporadic)
                    totalRelease += 1

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1


        # if len(stationQ) >= 2:
        #     stationQ = stationQ[stationQ[:, 1].argsort()]

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if availStation >= 1:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, realRelease  = popped[0], popped[1], popped[2], popped[3], popped[4], popped[5]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC, realRelease

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
                totalhighCnt += 1
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, realRelease2], dtype=np.int32).reshape(1,8)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        totalhighCnt += 1

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC, realRelease = runningStation[station]

                R_SW = time - releaseTime
                realR_SW = time - realRelease
                guaranteedRSW = RSW - releaseTime
                
                R_SW_list = np.append(R_SW_list, R_SW)
                realR_SW_list = np.append(realR_SW_list, realR_SW)
                RSW_list = np.append(RSW_list, guaranteedRSW)

                minusCharging = np.random.randint(0, taskSet[idx, _CG])
                realCharging = taskSet[idx, _CG] - minusCharging * dynamic

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, start, end, realRelease = runningCharger[charger, :]
                batterySet2[idx] += 1

                R_CG = time - releaseTime
                realR_CG = time - realRelease
                guaranteedRCG = RCG - releaseTime

                R_CG_list = np.append(R_CG_list, R_CG)
                realR_CG_list = np.append(realR_CG_list, realR_CG)
                RCG_list = np.append(RCG_list, guaranteedRCG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, Preemption, totalRelease, totalhighCnt, acceptCnt, R_SW_list, R_CG_list, realR_SW_list, realR_CG_list, RSW_list, RCG_list

@njit(fastmath=True)
def FIFOrunnerAHP2(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period, dynamic):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999999

    stationQ = np.empty((0, 6), dtype = np.int32)

    highQ = np.empty((0, 8), dtype = np.int32)
    lowQ = np.empty((0, 8), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 6), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 10), dtype = np.int32) * -1

    # measurement
    R_SW_list = np.empty((0), dtype = np.int64) #measured
    R_CG_list = np.empty((0), dtype = np.int64) #measured
    realR_SW_list = np.empty((0), dtype = np.int64) #measured
    realR_CG_list = np.empty((0), dtype = np.int64) #measured
    RSW_list = np.empty((0), dtype = np.int64) #guaranteed
    RCG_list = np.empty((0), dtype = np.int64) #guaranteed
    Preemption = 0
    totalRelease = 0
    totalhighCnt = 0
    acceptCnt = 0

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    TT = taskSet[idx, _T]
                    if time - prevRelease[idx] < TT:
                        a = np.vstack((stationQ, np.array([idx, prevRelease[idx] + TT, taskSet[idx, _D] + prevRelease[idx] + TT - 1, taskSet[idx, _RSW] + prevRelease[idx] + TT, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        nextRelease[idx] = prevRelease[idx] + TT
                        acceptCnt += 1
                    else:
                        a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        acceptCnt += 1
                    stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter
                    # nextRelease[idx] += (TT + sporadic)
                    nextRelease[idx] = time + (TT + sporadic)
                    totalRelease += 1

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1


        if len(stationQ) >= 2:
            stationQ = stationQ[stationQ[:, 1].argsort()]

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if stationQ[0][1] > time:
                break
            if availStation >= 1:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, realRelease  = popped[0], popped[1], popped[2], popped[3], popped[4], popped[5]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC, realRelease

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
                totalhighCnt += 1
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, realRelease2], dtype=np.int32).reshape(1,8)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        totalhighCnt += 1

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC, realRelease = runningStation[station]

                R_SW = time - releaseTime
                realR_SW = time - realRelease
                guaranteedRSW = RSW - releaseTime
                
                R_SW_list = np.append(R_SW_list, R_SW)
                realR_SW_list = np.append(realR_SW_list, realR_SW)
                RSW_list = np.append(RSW_list, guaranteedRSW)

                minusCharging = np.random.randint(0, taskSet[idx, _CG])
                realCharging = taskSet[idx, _CG] - minusCharging * dynamic

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, start, end, realRelease = runningCharger[charger, :]
                batterySet2[idx] += 1

                R_CG = time - releaseTime
                realR_CG = time - realRelease
                guaranteedRCG = RCG - releaseTime

                R_CG_list = np.append(R_CG_list, R_CG)
                realR_CG_list = np.append(realR_CG_list, realR_CG)
                RCG_list = np.append(RCG_list, guaranteedRCG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, Preemption, totalRelease, totalhighCnt, acceptCnt, R_SW_list, R_CG_list, realR_SW_list, realR_CG_list, RSW_list, RCG_list



@njit(fastmath=True)
def FIFOrunnerAHP3(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period, dynamic):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999999

    stationQ = np.empty((0, 6), dtype = np.int32)

    highQ = np.empty((0, 8), dtype = np.int32)
    lowQ = np.empty((0, 8), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 6), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 10), dtype = np.int32) * -1

    # measurement
    R_SW_list = np.empty((0), dtype = np.int64) #measured
    R_CG_list = np.empty((0), dtype = np.int64) #measured
    realR_SW_list = np.empty((0), dtype = np.int64) #measured
    realR_CG_list = np.empty((0), dtype = np.int64) #measured
    RSW_list = np.empty((0), dtype = np.int64) #guaranteed
    RCG_list = np.empty((0), dtype = np.int64) #guaranteed
    Preemption = 0
    totalRelease = 0
    totalhighCnt = 0
    acceptCnt = 0

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    TT = taskSet[idx, _T]
                    if time - prevRelease[idx] < TT:
                        a = np.vstack((stationQ, np.array([idx, prevRelease[idx] + TT, taskSet[idx, _D] + prevRelease[idx] + TT - 1, taskSet[idx, _RSW] + prevRelease[idx] + TT, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        nextRelease[idx] = prevRelease[idx] + TT
                        acceptCnt += 1
                    elif time - prevRelease[idx] > TT:
                        acceptCnt += 1
                        t_early_virt = RUNTIME + 1
                        qLen = len(stationQ)
                        for sQidx in range(qLen):
                            if stationQ[sQidx][1] >= prevRelease[idx] + TT and stationQ[sQidx][1] < time:
                                t_early_virt = stationQ[sQidx][1]
                                # print(1)
                                break
                        targetTime = min(time, t_early_virt)
                        a = np.vstack((stationQ, np.array([idx, targetTime, taskSet[idx, _D] + targetTime - 1, taskSet[idx, _RSW] + targetTime, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        nextRelease[idx] = targetTime
                    else:
                        a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C], time], dtype=np.int32).reshape(1,6)))
                        acceptCnt += 1
                    stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter
                    nextRelease[idx] = time + (TT + sporadic)
                    totalRelease += 1

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1


        if len(stationQ) >= 2:
            stationQ = stationQ[stationQ[:, 1].argsort()]

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if stationQ[0][1] > time:
                break
            if availStation >= 1:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, realRelease  = popped[0], popped[1], popped[2], popped[3], popped[4], popped[5]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC, realRelease

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
                totalhighCnt += 1
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease], dtype=np.int32).reshape(1,8)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end, realRelease2 = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, realRelease2], dtype=np.int32).reshape(1,8)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        totalhighCnt += 1

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, realRelease = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC, realRelease])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC, realRelease = runningStation[station]

                R_SW = time - releaseTime
                realR_SW = time - realRelease
                guaranteedRSW = RSW - releaseTime
                
                R_SW_list = np.append(R_SW_list, R_SW)
                realR_SW_list = np.append(realR_SW_list, realR_SW)
                RSW_list = np.append(RSW_list, guaranteedRSW)

                minusCharging = np.random.randint(0, taskSet[idx, _CG])
                realCharging = taskSet[idx, _CG] - minusCharging * dynamic

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realCharging, releaseTime, taskSet[idx, _RCG] + RSW, realRelease], dtype=np.int32).reshape(1,8)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, start, end, realRelease = runningCharger[charger, :]
                batterySet2[idx] += 1

                R_CG = time - releaseTime
                realR_CG = time - realRelease
                guaranteedRCG = RCG - releaseTime

                R_CG_list = np.append(R_CG_list, R_CG)
                realR_CG_list = np.append(realR_CG_list, realR_CG)
                RCG_list = np.append(RCG_list, guaranteedRCG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, Preemption, totalRelease, totalhighCnt, acceptCnt, R_SW_list, R_CG_list, realR_SW_list, realR_CG_list, RSW_list, RCG_list



@njit(fastmath=True)
def FIFOrunnerAHP4(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999999

    stationQ = np.empty((0, 5), dtype = np.int32)

    highQ = np.empty((0, 7), dtype = np.int32)
    lowQ = np.empty((0, 7), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 5), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 9), dtype = np.int32) * -1

    # measurement
    MAX_R_SW = np.zeros((numt), dtype = np.int32)
    MAX_R_CG = np.zeros((numt), dtype = np.int32)
    Preemption = 0

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    TT = taskSet[idx, _T]
                    t_early_virt = RUNTIME + 1
                    qLen = len(stationQ)
                    for sQidx in range(qLen):
                        if stationQ[sQidx][1] >= prevRelease[idx] and stationQ[sQidx][1] < time:
                            t_early_virt = stationQ[sQidx][1]
                            break
                    if t_early_virt != RUNTIME + 1:
                        targetTime = min(time, t_early_virt)
                    else:
                        if time - prevRelease[idx] < TT:
                            targetTime = prevRelease[idx] + TT
                        else:
                            targetTime = time
                    a = np.vstack((stationQ, np.array([idx, targetTime, taskSet[idx, _D] + targetTime - 1, taskSet[idx, _RSW] + targetTime, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                    nextRelease[idx] = targetTime
                    stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter
                    nextRelease[idx] = time + (TT + sporadic)

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1


        if len(stationQ) >= 2:
            stationQ = stationQ[stationQ[:, 1].argsort()]

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if availStation >= 0:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC  = popped[0], popped[1], popped[2], popped[3], popped[4]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2], dtype=np.int32).reshape(1,7)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC = runningStation[station]

                R_SW = time - releaseTime

                MAX_R_SW[idx] = max(MAX_R_SW[idx], R_SW)
                

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, taskSet[idx, _CG], releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, taskSet[idx, _CG], releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx = runningCharger[charger][0]
                batterySet2[idx] += 1

                releaseTime = runningCharger[charger][3] # t + RSW
                R_CG = time - releaseTime

                MAX_R_CG[idx] = max(MAX_R_CG[idx], R_CG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, MAX_R_SW, MAX_R_CG, Preemption

@njit(fastmath=True)
def FIFOrunner(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)

    stationQ = np.empty((0, 5), dtype = np.int32)

    highQ = np.empty((0, 7), dtype = np.int32)
    lowQ = np.empty((0, 7), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 5), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 9), dtype = np.int32) * -1



    # measurement
    MAX_R_SW = np.zeros((numt), dtype = np.int32)
    MAX_R_CG = np.zeros((numt), dtype = np.int32)
    Preemption = 0

    # idx, time, deadline, RSW, remainC

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                    stationQ = a
                    TT = taskSet[idx, _T]
                    sporadic = np.random.randint(0, TT + 1) * period
                    nextRelease[idx] += (TT + sporadic)

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if availStation >= 0:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC  = popped[0], popped[1], popped[2], popped[3], popped[4]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2], dtype=np.int32).reshape(1,7)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC = runningStation[station]

                R_SW = time - releaseTime

                MAX_R_SW[idx] = max(MAX_R_SW[idx], R_SW)
                

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, taskSet[idx, _CG], releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, taskSet[idx, _CG], releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx = runningCharger[charger][0]
                batterySet2[idx] += 1

                releaseTime = runningCharger[charger][3] # t + RSW
                R_CG = time - releaseTime

                MAX_R_CG[idx] = max(MAX_R_CG[idx], R_CG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, MAX_R_SW, MAX_R_CG, Preemption

@njit(fastmath=True)
def FIFOrunner_dynamic(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)

    stationQ = np.empty((0, 5), dtype = np.int32)

    highQ = np.empty((0, 7), dtype = np.int32)
    lowQ = np.empty((0, 7), dtype = np.int32)
    
    runningStation = np.ones((NUMP, 5), dtype = np.int32) * -1
    runningCharger = np.ones((NUMC, 9), dtype = np.int32) * -1

    np.random.seed(0)

    # measurement
    MAX_R_SW = np.zeros((numt), dtype = np.int32)
    MAX_R_CG = np.zeros((numt), dtype = np.int32)
    Preemption = 0

    # idx, time, deadline, RSW, remainC

    for time in range(RUNTIME):

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                    stationQ = a
                    TT = taskSet[idx, _T]
                    sporadic = np.random.randint(0, TT + 1) * period
                    nextRelease[idx] += (TT + sporadic)

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1

        qLen = len(stationQ)
        for sQidx in range(qLen):
            if availStation >= 0:
                for station in range(NUMP):
                    idx = stationQ[0, 0]
                    if stationCheck[station, time] == -1 and batterySet2[idx] >= 1:
                        popped, stationQ = stationQ[0, :], stationQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC  = popped[0], popped[1], popped[2], popped[3], popped[4]
                        stationCheck[station, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        runningStation[station] = idx, releaseTime, deadline, RSW, remainC

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3-0: low to high
        
        # currently running
        for charger in range(NUMC):
            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
            if idx2 != -1 and time == RSW2:
                chargerCheck[charger, time : end] = idx2
        
        # ready in lowQ
        qLen = len(lowQ)
        for i in range(qLen):
            popped, lowQ = lowQ[0, :], lowQ[1:, :]
            idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

            if time == RSW:
                # move to highQ
                b = np.vstack((highQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                highQ = b

            else:
                # back to lowQ
                b = np.vstack((lowQ, np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG], dtype=np.int32).reshape(1,7)))    
                lowQ = b

        # Step 3: run charger
        availHighCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] < 0:
                availHighCharger += 1

        # highQ sorting
        if len(highQ) >= 2:
            highQ = highQ[highQ[:, 3].argsort()] # sort with t + RSW

        # highQ
        qLen = len(highQ)
        for i in range(qLen):
            if availHighCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] < 0:
                        
                        # pop highQ
                        popped, highQ = highQ[0, :], highQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        if chargerCheck[charger, time] != -1:
                            # preemption, low go back to lowQ
                            idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2, start, end = runningCharger[charger, :]
                            chargerCheck[charger, time : end] = -1
                            remainC2 = remainC2 - (time - start)
                            b = np.vstack((lowQ, np.array([idx2, releaseTime2, deadline2, RSW2, remainC2, initRelease2, RCG2], dtype=np.int32).reshape(1,7)))    
                            lowQ = b

                            Preemption += 1

                        chargerCheck[charger, time : time + remainC] = idx
                        availHighCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break
                
        # lower
        availLowCharger = 0
        for charger in range(NUMC):
            if chargerCheck[charger, time] == -1:
                availLowCharger += 1

        qLen = len(lowQ)
        for i in range(qLen):
            if availLowCharger >= 0:
                for charger in range(NUMC):
                    if chargerCheck[charger, time] == -1:

                        # pop lowQ
                        popped, lowQ = lowQ[0, :], lowQ[1:, :]
                        idx, releaseTime, deadline, RSW, remainC, initRelease, RCG = popped

                        chargerCheck[charger, time : time + remainC] = -1000 - idx
                        availLowCharger -= 1
                        runningCharger[charger, :] = np.array([idx, releaseTime, deadline, RSW, remainC, initRelease, RCG, time, time+remainC])

                        if time + remainC > deadline + 1:
                            print("Fail")
                        assert time + remainC <= deadline + 1, "Fail"

                        break

        # Step 4: finish station (release battery)
        for station in range(NUMP):
            if stationCheck[station, time] != -1 and stationCheck[station, time + 1] == -1:
                idx = stationCheck[station, time]

                idx, releaseTime, deadline, RSW, remainC = runningStation[station]

                R_SW = time - releaseTime

                MAX_R_SW[idx] = max(MAX_R_SW[idx], R_SW)
                
                realChargingTime = np.random.randint(1, taskSet[idx, _CG] + 1)

                # insert it to highQ
                if time >= RSW:
                    # idx, releaseTime, deadline, RSW, remainC, initRelease, RCG
                    b = np.vstack((highQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realChargingTime, releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    highQ = b

                # insert it to lowQ
                else:
                    b = np.vstack((lowQ, np.array([idx, time, releaseTime + RSW + taskSet[idx, _VD] - 1, RSW, realChargingTime, releaseTime, taskSet[idx, _RCG] + time], dtype=np.int32).reshape(1,7)))    
                    lowQ = b
            
                runningStation[station] = np.array([-1,-1,-1,-1,-1])

        # Step 5: finish charger (battery num ++)
        for charger in range(NUMC):
            if chargerCheck[charger, time] != -1 and chargerCheck[charger, time + 1] == -1:
                idx = runningCharger[charger][0]
                batterySet2[idx] += 1

                releaseTime = runningCharger[charger][3] # t + RSW
                R_CG = time - releaseTime

                MAX_R_CG[idx] = max(MAX_R_CG[idx], R_CG)

                runningCharger[charger] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return stationCheck, chargerCheck, MAX_R_SW, MAX_R_CG, Preemption