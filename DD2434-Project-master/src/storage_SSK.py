import numpy as np
from math import sqrt


class KernelOperations:

    def __init__(self, S, T, N):
        self.S = np.array(list(S))
        self.T = np.array(list(T))

        # greatest substring length N
        self.N = N

        self.abs_S = len(S) + 1
        self.abs_T = len(T) + 1

        # K prime matrix
        self.Kp = np.zeros((N, self.abs_S, self.abs_T))
        self.Kp[0, :, :] = 1

        # K double prime matrix
        self.Kpp = np.zeros((N, self.abs_S, self.abs_T))

        # decay factor - penalizes non-contiguous substrings, value between 0 and 1
        self.lam = 0.5

        # list to store kernel values for different values of n
        self.kernel_values = []

    def run_all_kernels(self):
        for n in range(self.N):
            res = self.kernel(n)
            self.kernel_values.append(res)
        return self.kernel_values

    def kernel(self, n):
        self.build_layer(n)
        val = self.get_kernel(n)
        return val

    def build_layer(self, n):
        """ Kernel that gives the sum over all common subsequences
        weighted according to their frequency and length
        s, t = strings to be compared
        n = length of substrings """
        if min(len(self.S), len(self.T)) < n:
            return 0

        # iterate over all length of substrings
        for i in range(1, n):

            # iterate over every letter in s
            for s in range(1, self.abs_S):

                # iterate over every letter in t
                for t in range(1, self.abs_T):

                    if min(s, t) >= i:
                        # if last letter in T equals last letter in S
                        if self.S[s - 1] == self.T[t - 1]:
                            self.Kpp[i][s][t] = self.lam * (
                                    self.Kpp[i][s][t - 1] + self.lam * self.Kp[i - 1][s - 1][t - 1])

                        else:
                            tj = np.where(self.T[:t] == self.S[s - 1])[0]
                            self.Kpp[i][s][t] = np.sum(self.Kp[i - 1][s - 1][tj] * self.lam ** (t - tj + 1))

                        self.Kp[i][s][t] = self.lam * self.Kp[i][s - 1][t] + self.Kpp[i][s][t]

    def get_kernel(self, n):


        # K matrix
        K = np.zeros((self.abs_S, self.abs_T))

        # build the final kernel backwards instead of recursively
        for s in range(1, self.abs_S):
            for t in range(1, self.abs_T):

                if min(s, t) >= n:
                    tj = np.where(self.T[:t] == self.S[s - 1])[0]
                    K[s][t] = K[s - 1][t] + np.sum(self.Kp[n - 1][s - 1][tj] * self.lam ** 2)

        # return the final kernel from the full strings
        return K[-1][-1]

def main():
    S = "cells interlinked within cells interlinked"
    T = "within one stem and dreadfully distinct"
    N = 7

    ko = KernelOperations(S, T, N)

    ko.run_all_kernels()

    print(ko.kernel_values)


if __name__ == '__main__':
    main()