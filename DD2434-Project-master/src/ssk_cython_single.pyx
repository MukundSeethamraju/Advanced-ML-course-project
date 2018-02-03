#!/usr/bin/env python

import cython
import numpy as np
cimport numpy as cnp
cimport cython
from cpython cimport array

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _kernel(array.array S, array.array T, int n, double lam):
    cdef int len_S = len(S)
    cdef int len_T = len(T)
    cdef int abs_S = len_S + 1
    cdef int abs_T = len_T + 1
    # iteration variables
    cdef Py_ssize_t i, s, t, j
    cdef double the_sum = 0
    cdef cnp.ndarray [double, ndim=1] kernel_vals = np.zeros((n + 1), dtype=np.float)

    # K prime matrix
    cdef cnp.ndarray [double, ndim=3] Kp = np.zeros((n, abs_S, abs_T), dtype=np.float)
    Kp[0] = np.ones((abs_S, abs_T), dtype=np.float)

    cdef cnp.ndarray [long, ndim=1] tj = np.full((len_T), -1, dtype=np.int)
    cdef int tj_current_index = 0
    cdef int tj_ = 0

    # K double prime
    cdef double kpp = 0

    cdef cnp.ndarray [double, ndim=2] K = np.zeros((abs_S, abs_T), dtype=np.float)


    if int_min(len_S, len_T) < n:
        return kernel_vals
    # iterate over all length of substrings
    for i in range(1, n):
        # iterate over every letter in s
        for s in range(i, abs_S): # could probably kpp = 0
            tj = np.full((len_T), -1, dtype=np.int)
            tj_current_index = 0
            kpp = 0
            # iterate over every letter int 
            for t in range(i, abs_T):
                # if last letter in T equals last letter in S
                if S[s - 1] == T[t - 1]: # kpp = lam * (kpp + lam Kp[i -1][s -1][t -1])
                    tj[tj_current_index] = t-1
                    tj_current_index += 1
                    kpp = lam * (kpp + lam * Kp[i - 1][s - 1][t - 1])
                else:
                    the_sum = 0
                    for j in range(len_T):
                       tj_ = tj[j]
                       if tj_ == -1:
                           break
                       the_sum += Kp[i - 1][s - 1][tj_] * lam**(t - tj_ + 1)
                    kpp = the_sum
                Kp[i][s][t] = lam * Kp[i][s - 1][t] + kpp

    # Final step
    # build the all kernels backwards instead of recursively
    K = np.zeros((abs_S, abs_T), dtype=np.float)
    for s in range(1, abs_S):
        tj = np.full((len_T), -1, dtype=np.int)
        tj_current_index = 0
        for t in range(1, abs_T):
            if int_min(s ,t) >= n:
                if(T[t - 1] == S[s - 1]):
                    tj[tj_current_index] = t - 1
                    tj_current_index += 1
                the_sum = 0
                for j in range(len_T):
                    tj_ = tj[j]
                    if tj_ == -1:
                        break
                    the_sum  += Kp[i - 1][s - 1][tj_] * lam**2
                K[s][t] = K[s - 1][t] + the_sum
    
    return K[len_S][len_T]


# This function should probably be done during preprocessing
def get_arrays(s = "cells interlinked within cells interlinked", t="within one stem and dreadfully distinct"):
    S = array.array('l', [ord(character) for character in s])
    T = array.array('l', [ord(character) for character in t])
    return S,T
    

def kernel(s, t, n=3, lam=0.5):
    S, T = get_arrays(s, t)
    res = _kernel(S, T, n, lam)
    return res
