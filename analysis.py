import numpy as np
import operator
import copy
from numba import jit, njit
# period, wcet, deadline, gumbel, power, release, seed, slack, fake wcet
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

"""
NWCF : Non-Work-Conserving Factor
It is utilized to delay the execution of each jobs to reduce peak power
"""
############# Rate Monotonic ############# 

## Effective Analysis for Engineering Real-Time Fixed Priority Schedulers by Alan Burns, Ken Tindell, and Andy Wellings
## Non-Preemptive and Limited Preemptive Scheduling by Prof. Dr. Jian-Jia Chen, Georg von der Br¨uggen

# @jit
def oneAnalysisRM(oneSet): # one EDF analysis
    numTask = oneSet.shape[0]

    T = oneSet[:, _T]
    # T = T.reshape(1, T.size)

    C = oneSet[:, _C]
    # C = C.reshape(1, C.size)

    sortOrder = T.argsort()

    # T = T[0, sortOrder]
    # C = C[0, sortOrder]

    T = T[sortOrder]   
    C = C[sortOrder]   


    priority = T * 1000000- oneSet[:, _C] * 1000 + oneSet[:, _ID] # 똑같은 period인 task끼리도 priority가 fix되야 analysis가 유효함. C 큰 놈한테 priority 조금이라도 더 주는게 대강 더 유리함. C도 똑같으면 id로 구분

    if (C/T).sum() > 1: return True

    for i in range(numTask):
        hp = np.where(priority <= priority[i])[0]
        lp = np.where(priority > priority[i])[0]
        hp = hp[hp!=i]

        Ci = C[i]
        # Bi = max(np.append(C[lp], 0))
        Bi = (np.append(C[lp], 0)).max()
        Ti = T[i]

        Rprev = Ci + Bi
        while(True):
            Rnext = Ci + Bi + (np.ceil(Rprev/T[hp]) * C[hp]).sum()
            if Rnext > Ti:
                return True
            if Rnext == Rprev:
                break
            Rprev = Rnext


    return False


# @jit
def analysisRM(taskSet, NUMP):

    return 0

    nonSched = False

    for j in range(NUMP):
        
        oneSet = taskSet[j, :, :]
        nonSched = nonSched | oneAnalysisRM(oneSet)

    if nonSched == True:
        return -1
    else: 
        return 0


@jit
def makeOneNWCF_RM(oneSet, sort1, sort2, order1, order2): # _P largest, without round robin
    newSet = oneSet.copy()
    newSet[:, _M] = newSet[:, _D] - newSet[:, _C]
    
    one, two = newSet.shape
    # (newSet[:, _P] + (newSet[:, _P].max() + newSet[:, _P].min())/2)
    # newSet = np.concatenate((newSet, (newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]) / np.maximum(newSet[:, _F], np.ones(one)) ).reshape(one,1)), axis=-1)
    # newSet = np.concatenate((newSet, (newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]) / newSet[:, _C] ).reshape(one,1)), axis=-1)
    newSet = np.concatenate((newSet, (newSet[:, _F] / (newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]))).reshape(one,1)), axis=-1)
    newSet = np.concatenate((newSet, (newSet[:, _P] * newSet[:, _C] ).reshape(one,1)), axis=-1)

    newSet = newSet[newSet[:,sort1].argsort(kind = 'mergesort')[::order1]]
    newSet = newSet[newSet[:,sort2].argsort(kind = 'mergesort')[::order2]]

    while True:    
        change = 0
        for k in range(newSet.shape[0]):
            i = newSet[k, :]

            i[_C] += 1
            i[_F] += 1
            tempSet = newSet.copy()

            if oneAnalysisRM(tempSet) == False: 
                change = 1
                newSet = tempSet
                newSet[:, _M] = newSet[:, _D] - newSet[:, _C]
                # newSet[:, _PF] = newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]) / np.maximum(newSet[:, _F], np.ones(one))
                # newSet[:, _PF] = newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]) / newSet[:, _C]
                newSet[:, _PF] = newSet[:, _F] / (newSet[:, _P] * (newSet[:, _C] - newSet[:, _F]))
                newSet = newSet[newSet[:,sort1].argsort(kind = 'mergesort')[::order1]]
                newSet = newSet[newSet[:,sort2].argsort(kind = 'mergesort')[::order2]]
                break
                   
            else:
                i[_C] -= 1
                i[_F] -= 1
        if change == 0:
            break
    
    return newSet[:, :11]

@jit
def makeNWCF_RM(taskSet, NUMP, sort1, sort2, order1, order2):
    newTaskSet = copy.deepcopy(taskSet)

    for j in range(NUMP):
        oneSet = taskSet[j, :, :]
        numTask = oneSet.shape[0]
        newTaskSet[j, :, :] = makeOneNWCF_RM(oneSet, sort1, sort2, order1, order2)
    
    return newTaskSet