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
_CG = 8


@njit(fastmath=True)
def FIFOrunnerAHP1(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 5), dtype = np.int32)
    semiChargerReadyQ = np.empty((0, 3), dtype = np.int32)

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
                    if time - prevRelease[idx] < TT:
                        continue
                    else:
                        a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                    stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter 
                    nextRelease[idx] += (TT + sporadic)

        # Step 2: run station (battery num --)
        availStation = 0
        for station in range(NUMP):
            if stationCheck[station, time] == -1:
                availStation += 1


        # if len(stationQ) >= 2:
        #     stationQ = stationQ[stationQ[:, 1].argsort()]

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
def FIFOrunnerAHP2(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM, period):
    taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    # taskSet = paramTaskSet

    batterySet2 = batterySet.copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)
    prevRelease = np.ones((numt), dtype = np.int32) * -9999

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 5), dtype = np.int32)
    semiChargerReadyQ = np.empty((0, 3), dtype = np.int32)

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
                    if time - prevRelease[idx] < TT:
                        a = np.vstack((stationQ, np.array([idx, prevRelease[idx] + TT, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                        nextRelease[idx] = prevRelease[idx] + TT
                    else:
                        a = np.vstack((stationQ, np.array([idx, time, taskSet[idx, _D] + time - 1, taskSet[idx, _RSW] + time, taskSet[idx, _C]], dtype=np.int32).reshape(1,5)))
                    stationQ = a
                    prevRelease[idx] = nextRelease[idx]
                    sporadic = np.random.randint(-TT/2, TT/2 + 1) * period # jitter
                    nextRelease[idx] += (TT + sporadic)

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

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 5), dtype = np.int32)
    semiChargerReadyQ = np.empty((0, 3), dtype = np.int32)

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