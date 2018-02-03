import cython
cimport cython
import numpy as np
cimport numpy as np

cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] _kernel(long[:] S, long[:] T, int n, double lam):


    cdef int len_S = len(S)
    cdef int len_T = len(T)
    cdef int abs_S = len_S + 1
    cdef int abs_T = len_T + 1

    # iteration variables
    cdef int i, s, t, j

    # Result array
    cdef double [:] kernel_vals = np.zeros((n + 1))
    kernel_vals[:] = 0

    # K prime matrix
    cdef double[:,:,:] Kp = np.zeros((n, abs_S, abs_T))
    Kp[0,:,:] = 1

    # tj array
    cdef int[:] tj = np.zeros((len_T), dtype=np.int32)
    tj[:] = -1
    cdef int tj_current_index = 0
    cdef int tj_ = 0

    # K biss
    cdef double kpp = 0

    cdef double the_sum = 0
    
    # K
    cdef double[:,:] K = np.zeros((abs_S, abs_T))


    if int_min(len_S, len_T) < n:
        return kernel_vals
    # iterate over all length of substrings
    for i in range(1, n):
        # iterate over every letter in s
        for s in range(i, abs_S):
            # iterate over every letter int 
            for t in range(i, abs_T):
                # if last letter in T equals last letter in S
                if S[s - 1] == T[t - 1]: # kpp = lam * (kpp + lam Kp[i -1][s -1][t -1])
                    tj[tj_current_index] = t - 1
                    tj_current_index += 1
                    kpp = lam * (kpp + lam * Kp[i - 1, s - 1, t - 1])
                else:
                    for j in range(len_T):
                       tj_ = tj[j]
                       if tj_ == -1:
                           break
                       the_sum += Kp[i - 1, s - 1, tj_] * lam**(t - tj_ + 1)
                    kpp = the_sum
                    the_sum = 0
                Kp[i, s, t] = lam * Kp[i, s - 1, t] + kpp
            # Reset variables
            tj[:] = -1
            tj_current_index = 0
            kpp = 0

    # Final step
    # build the all kernels backwards instead of recursively
    for i in range(1,n + 1):
        for s in range(i, abs_S):
            for t in range(i, abs_T):
                if(T[t - 1] == S[s - 1]):
                    tj[tj_current_index] = t - 1
                    tj_current_index += 1
                for j in range(len_T):
                    tj_ = tj[j]
                    if tj_ == -1:
                        break
                    the_sum  += Kp[i - 1, s - 1, tj_] * lam**2
                K[s, t] = K[s - 1, t] + the_sum
                the_sum = 0
            tj[:] = -1
            tj_current_index = 0
        kernel_vals[i] = K[len_S, len_T]
        K[:] = 0
    return kernel_vals


def get_arrays(s = "cells interlinked within cells interlinked", t = "within one stem and dreadfully distinct"):
    S = np.array([ord(character) for character in s], dtype=np.int)
    T = np.array([ord(character) for character in t], dtype=np.int)    
    return S,T


def get_array(s):
    S = np.array([ord(character) for character in s], dtype=np.int)
    return S
    

def kernel(s, t, n=3, lam=0.5):
    res = _kernel(s, t, n, lam)
    return res


def kernel_string(S, T, n, lam):
    s,t = get_arrays(S,T)
    res = _kernel(s,t,n,lam)
    return res


def tests():
    np.testing.assert_array_equal(kernel_string('car', 'car', 3, 0.5), np.array([0, 0.75, 0.140625, 0.015625]))
    np.testing.assert_array_equal(kernel_string('cat', 'car', 2, 0.5), np.array([0, 0.5, 0.0625])) 
    np.testing.assert_array_almost_equal(kernel_string('cells interlinked within cells interlinked', 'within one stem and dreadfully distinct', 6, 0.5), np.array([0, 3.10000000e+01, 2.90871475e+00, 3.05520163e-01, 4.44025374e-02, 7.64298965e-03, 1.14451672e-03]))


if __name__ == '__main__':
    tests()
