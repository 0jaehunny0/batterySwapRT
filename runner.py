from asyncio import tasks
from numba import jit, njit, vectorize
import numpy as np
import copy

# from sklearn.utils import shuffle

_T = 0
_C = 1 #  C^SW
_D = 2
_ID = 3
_CG = 4 # C^CG


def FIFOrunnerLeakyQueue(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM):
    # taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    taskSet = paramTaskSet
    taskSet = np.hstack((taskSet, np.array([C_CG]).T))
    batterySet2 = np.array(batterySet).copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 4), dtype = np.int32)

    leakyQ = []
    for types in range(numt):
        leakyQ.append(np.empty((0, 4), dtype = np.int32))
    leakyQRelease = np.zeros((numt), dtype = np.int32)

    for time in range(RUNTIME):
        print(1)

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    b = np.vstack((stationReadyQ, np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                    stationReadyQ = b
                    nextRelease[idx] += taskSet[idx, _T]


        # Step 2: run station (battery num --)
        availStation = 0
        for i in range(NUMP):
            if stationCheck[i, time] == -1:
                availStation += 1


        qLen = len(stationReadyQ)
        for i in range(qLen):
            if availStation >= 0:
                for j in range(NUMP):
                    if stationCheck[j, time] == -1 and batterySet2[idx] >= 0:
                        popped, stationReadyQ = stationReadyQ[0, :], stationReadyQ[1:, :]
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        stationCheck[j, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3: run charger
        availCharger = 0
        for i in range(NUMC):
            if chargerCheck[i, time] == -1:
                availCharger += 1

        for i in range(numt):
            if availCharger >= 0 and len(leakyQ[i]) > 0 and leakyQRelease[i] <= time:
                for j in range(NUMC):
                    if chargerCheck[j, time] == -1:
                        popped = leakyQ[i][0, :]
                        c = np.copy(leakyQ[i][1:, :])
                        leakyQ[i] = c
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        chargerCheck[j, time : time + taskSet[idx, _CG]] = idx
                        availCharger -= 1
                        leakyQRelease[i] = time + taskSet[i, _T]
                        break
        
        # Step 4: finish station (release battery)
        for i in range(NUMP):
            if stationCheck[i, time] != -1 and stationCheck[i, time + 1] == -1:
                idx = stationCheck[i, time]
                b = np.vstack((leakyQ[idx], np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                leakyQ[idx] = b
        
        # Step 5: finish charger (battery num ++)
        for i in range(NUMC):
            if chargerCheck[i, time] != -1 and chargerCheck[i, time + 1] == -1:
                idx = chargerCheck[i, time]
                batterySet2[idx] += 1

        print(1)

    return stationCheck, chargerCheck

@njit(fastmath=True)
def FIFOrunner(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM):
    # taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    taskSet = paramTaskSet
    taskSet = np.hstack((taskSet, np.array([C_CG]).T))

    batterySet2 = np.array(batterySet).copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 4), dtype = np.int32)

    for time in range(RUNTIME):
        print(1)

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    b = np.vstack((stationReadyQ, np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                    stationReadyQ = b
                    nextRelease[idx] += taskSet[idx, _T]


        # Step 2: run station (battery num --)
        availStation = 0
        for i in range(NUMP):
            if stationCheck[i, time] == -1:
                availStation += 1

        qLen = len(stationReadyQ)
        for i in range(qLen):
            if availStation >= 0:
                for j in range(NUMP):
                    if stationCheck[j, time] == -1 and batterySet2[idx] >= 0:
                        popped, stationReadyQ = stationReadyQ[0, :], stationReadyQ[1:, :]
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        stationCheck[j, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3: run charger
        availCharger = 0
        for i in range(NUMC):
            if chargerCheck[i, time] == -1:
                availCharger += 1

        qLen = len(chargerReadyQ)
        for i in range(qLen):
            if availCharger >= 0:
                for j in range(NUMC):
                    if chargerCheck[j, time] == -1:
                        popped, chargerReadyQ = chargerReadyQ[0, :], chargerReadyQ[1:, :]
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        chargerCheck[j, time : time + taskSet[idx, _CG]] = idx
                        availCharger -= 1
                        break
        
        # Step 4: finish station (release battery)
        for i in range(NUMP):
            if stationCheck[i, time] != -1 and stationCheck[i, time + 1] == -1:
                idx = stationCheck[i, time]
                b = np.vstack((chargerReadyQ, np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                chargerReadyQ = b
        
        # Step 5: finish charger (battery num ++)
        for i in range(NUMC):
            if chargerCheck[i, time] != -1 and chargerCheck[i, time + 1] == -1:
                idx = chargerCheck[i, time]
                batterySet2[idx] += 1

        print(1)

    return stationCheck, chargerCheck
    
def FIFOrunner2(paramTaskSet, NUMP, RUNTIME, batterySet, C_CG, chargerNUM):
    # taskSet = paramTaskSet.copy()

    NUMC = chargerNUM
    taskSet = paramTaskSet
    taskSet = np.hstack((taskSet, np.array([C_CG]).T))

    batterySet2 = np.array(batterySet).copy()

    stationCheck = np.ones( (NUMP, RUNTIME+1), dtype=np.int32) * -1
    chargerCheck = np.ones( (NUMC, RUNTIME+1), dtype=np.int32) * -1

    numt = taskSet.shape[0]
    nextRelease = np.zeros((numt), dtype = np.int32)

    stationReadyQ = np.empty((0, 4), dtype = np.int32)
    chargerReadyQ = np.empty((0, 4), dtype = np.int32)

    for time in range(RUNTIME):
        print(1)

        # Step 1: car release
        a = nextRelease.min()
        if a <= time:
            for idx in range(numt):
                if nextRelease[idx] == time and nextRelease[idx] + taskSet[idx, _D] - 1 < RUNTIME:
                    b = np.vstack((stationReadyQ, np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                    stationReadyQ = b
                    nextRelease[idx] += taskSet[idx, _T]


        # Step 2: run station (battery num --)
        availStation = 0
        for i in range(NUMP):
            if stationCheck[i, time] == -1:
                availStation += 1

        qLen = len(stationReadyQ)
        for i in range(qLen):
            if availStation >= 0:
                for j in range(NUMP):
                    if stationCheck[j, time] == -1 and batterySet2[idx] >= 0:
                        popped, stationReadyQ = stationReadyQ[0, :], stationReadyQ[1:, :]
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        stationCheck[j, time : time + taskSet[idx, _C]] = idx
                        availStation -= 1
                        batterySet2[idx] -= 1 # minus battery num

                        if time + taskSet[idx, _C] > deadline + 1:
                            print("Fail")
                        assert time + taskSet[idx, _C] <= deadline + 1, "Fail"

                        break
            else:
                break
    
        # Step 3: run charger
        availCharger = 0
        for i in range(NUMC):
            if chargerCheck[i, time] == -1:
                availCharger += 1

        qLen = len(chargerReadyQ)
        for i in range(qLen):
            if availCharger >= 0:
                for j in range(NUMC):
                    if chargerCheck[j, time] == -1:
                        popped, chargerReadyQ = chargerReadyQ[0, :], chargerReadyQ[1:, :]
                        idx, releaseTime, deadline, zero  = popped[0], popped[1], popped[2], popped[3]
                        chargerCheck[j, time : time + taskSet[idx, _CG]] = idx
                        availCharger -= 1
                        break
        
        # Step 4: finish station (release battery)
        for i in range(NUMP):
            if stationCheck[i, time] != -1 and stationCheck[i, time + 1] == -1:
                idx = stationCheck[i, time]
                b = np.vstack((chargerReadyQ, np.array([idx, time, taskSet[idx, _D] + time - 1, 0], dtype=np.int32).reshape(1,4)))
                chargerReadyQ = b
        
        # Step 5: finish charger (battery num ++)
        for i in range(NUMC):
            if chargerCheck[i, time] != -1 and chargerCheck[i, time + 1] == -1:
                idx = chargerCheck[i, time]
                batterySet2[idx] += 1

        print(1)

    return stationCheck, chargerCheck

@njit(fastmath=True)
def partitionedRunner3(paramTaskSet, NUMP, RUNTIME, OPTS, MODE, WHAT, ORDER):
    
    taskSet = paramTaskSet.copy()
    for partition in range(NUMP):
        taskSet[partition, :, :] = taskSet[partition, :, :][taskSet[partition, :, :][:,_ID].argsort()[::1]]    
    
    tempSave = taskSet[0, :, :].shape[0]

    nextRelease = np.zeros((NUMP, tempSave), dtype = np.int32)

    readyQueue = []
    for partition in range(NUMP):
        readyQueue.append(np.empty((0, 3), dtype = np.int32))
    semiRunningQueue = np.empty((0, 7), dtype = np.int32)

    powerCheck = np.zeros(RUNTIME, dtype = np.int32)
    pseudoPowerCheck = np.zeros(RUNTIME, dtype = np.int32)
    runningCheck = np.ones( (NUMP, RUNTIME), dtype=np.int32) * -1

    # wakeUpList = []

    priority = taskSet[:, :, _T] * 1000000 - taskSet[:, :, _C]*1000 + taskSet[:, :, _ID]

    for time in range(RUNTIME):
        
        flag = False

        #semi Running Queue는 reserve 해놓은 애들 중 실행 시작 안 한 애들
        semiRunningQueue = semiRunningQueue[semiRunningQueue[:,4].argsort()[::1]] 
        saving = semiRunningQueue.shape[0]
        for temp in range(saving): 

            check = semiRunningQueue[0][4]
            tempTask = semiRunningQueue[0]
            if check == time:
                popped, semiRunningQueue = semiRunningQueue[0, :], semiRunningQueue[1:, :]
                idx, partition = popped[0], popped[2]
                # actualEnd = np.random.randint(1, 1 + (taskSet[partition, idx, _C] - taskSet[partition, idx, _F]))
                # wakeUpList.append(time + popped[6])
                powerCheck[time : time + popped[6]] += np.int32(taskSet[partition, idx, _P])
            else:
                break

        # periodic하게 release 되는 애들
        a = nextRelease.min()
        if a <= time:
            for partition in range(NUMP):
                for idx in range(tempSave):
                    if nextRelease[partition, idx] == time and nextRelease[partition, idx] + taskSet[partition, idx, _D] - 1 < RUNTIME:
                        b = np.vstack((readyQueue[partition], np.array([idx, taskSet[partition, idx, _D]  + time - 1, priority[partition, idx]], dtype=np.int32).reshape(1, 3) ))  # RM same period 보정
                        
                        readyQueue[partition] = b
                        nextRelease[partition, idx] += taskSet[partition, idx, _T]


        # RM 따라서 release 한 애들 중 semi running queue에 넣어 reserve 해줄 애들 선정 (현재 idle한 partition이고 레디 큐가 차있는 경우)
        for partition in range(NUMP):
            if runningCheck[partition, time] == -1 and len(readyQueue[partition]) != 0:
                readyQueue[partition] = readyQueue[partition][readyQueue[partition][:,2].argsort()[::1]] # [readyQueue[partition][:,1] 에 prioity 저장하면 됨
                popped = readyQueue[partition][0,:]
                c = np.copy(readyQueue[partition][1:,:])
                readyQueue[partition] = c
                idx = popped[0]
                np.random.seed(popped[1]+partition+idx)
                # actualEnd = np.round_((taskSet[partition, idx, _C] - taskSet[partition, idx, _F])/2) + 1
                # actualEnd = np.random.randint(1, 1 + (taskSet[partition, idx, _C] - taskSet[partition, idx, _F]))
                aa = -5.1993375821928165 # probability < 0.0000001 
                bb = 5.1993375821928165 
                pb = np.random.normal()
                WCET = taskSet[partition, idx, _C] - taskSet[partition, idx, _F]
                if pb < aa or pb > bb:
                    pb = np.random.normal()
                actualEnd = np.round_( (pb - aa) / (bb - aa) * (WCET - 1) ) + 1
                # norm magic number -3.719016485455709, 3.719016485455709
                # gumbel magic number -2.2203268063678463, 9.210290369892835
                # Euler–Mascheroni constant: 0.5772 (mean of gumbel distrubution)
                dd = np.concatenate((popped[:2], np.array([partition, taskSet[partition, idx, WHAT], -1,  time, actualEnd])))
                e = dd.reshape(1, 7) 
                f = np.copy(e)
                g = f.astype(np.int32)
                semiRunningQueue = np.vstack((semiRunningQueue, g))
                runningCheck[partition, time : time + taskSet[partition, idx, _C]] = idx

                if time + taskSet[partition, idx, _C] > popped[1] + 1:
                    print("fail")

                assert time + taskSet[partition, idx, _C] <= popped[1] + 1, "Fail"
                flag = True
        
        # semi running queue에 새로 넘어온 애가 있는 경우 언제 actual execution 시작할지 결정
        # WHAT은 semi running queue actual execution 시작하는 순서 뭐를 기준으로 할 것이냐 결정

        # for xx in range(len(wakeUpList)):
        #     yy = wakeUpList.pop(0)
        #     if yy == time:
        #         flag = True
        #     else:
        #         wakeUpList.append(yy)

        if flag == True:
            for track in range(time, min(RUNTIME, time + 101)):
                pseudoPowerCheck[track] = np.int32(powerCheck[track] )
            if WHAT == _F:
                semiRunningQueue = semiRunningQueue[(semiRunningQueue[:,3] + semiRunningQueue[:,5]).argsort()[::ORDER]] 
            elif WHAT == _D:
                semiRunningQueue = semiRunningQueue[semiRunningQueue[:,1] .argsort()[::ORDER]]
            elif WHAT == _ID:
                semiRunningQueue = semiRunningQueue[semiRunningQueue[:,2] .argsort()[::ORDER]]
            else:
                semiRunningQueue = semiRunningQueue[semiRunningQueue[:,3] .argsort()[::ORDER]]
            saving = semiRunningQueue.shape[0]
            for temp in range(saving):
                popped, semiRunningQueue = semiRunningQueue[0, :], semiRunningQueue[1:, :]
                idx, partition = popped[0], popped[2]
                remainNWCF = taskSet[partition, idx, _F] - time + popped[5]
                check = 0
                if check == 0:
                    # actualEnd = np.random.randint(1, 1 + (taskSet[partition, idx, _C] - taskSet[partition, idx, _F]))
                    # wakeUpList.append(time + popped[6])
                    powerCheck[time : time + popped[6]] += np.int32(taskSet[partition, idx, _P])
                else:
                    popped[4] = time + check
                    i = np.copy(popped)
                    semiRunningQueue = np.vstack((semiRunningQueue, i.reshape(1,7)))
                
    return powerCheck
