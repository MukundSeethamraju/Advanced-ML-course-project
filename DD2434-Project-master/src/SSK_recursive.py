# SSK kernel - uses recursion

from math import sqrt

lam = 0.5

def K_recursive(sx, t, n):
    """Kernel that gives the sum over all common sub sequences weighted according to their frequency and length
    sx, t = string 1, 2
    n = max length of sub strings"""

    # stop recursion condition
    if min(len(sx), len(t)) < n:
        return 0

    # x is the last character in string sx
    x = sx[-1]
    # s is the string s without x
    s = sx[0: -1]

    # performs summation of K(n-1) prime
    sum = 0
    for j, tj in enumerate(t):
        if tj == x:
            sum += K_prime(s, t[0: j], n-1) * (lam ** 2)

    ret = K_recursive(s, t, n) + sum

    # print("K ret: ", ret)

    return ret


def K_prime(sx, t, i):
    """aiding function that:
    counts the length from the beginning of the the particular sequence through the end of string s and t
    instead of just counting the lengths"""
    if i == 0:
        return 1
    if min(len(sx), len(t)) < i:
        return 0

    s = sx[0: -1]

    ret = lam * K_prime(s, t, i) + K_double_prime(sx, t, i)
    # print("K_prime ret: ", ret)

    return ret


def K_double_prime(sx, t, i):
    """function to speed up calculation of K_prime_i"""
    if min(len(sx), len(t)) < i:
        return 0

    x = sx[-1]
    s = sx[0:-1]

    # if the two string share the same last letter, faster calculation
    if x == t[-1]:
        t = t[0: -1]

        ret = lam * (K_double_prime(sx, t, i) + lam * K_prime(s, t, i-1))

        # print("K double prime ret: ", ret)

        return ret

    # else perform the original K'' calculation
    else:
        sum = 0
        for j, tj in enumerate(t):
            if tj == x:
                sum += K_prime(s, t[0: j], i-1) * lam ** (len(t) - j + 1)

        # print("K double prime ret: ", sum)
        return sum



def K_norm(s, t, n):
    print("\nstrings: ")
    print("s = ", s)
    print("t = ", t)
    print("\nfirst kernel executing...")
    k1 = K_recursive(s, s, n)
    print("kernel(s, s, n) ", k1)
    print("done")
    print("\nsecond kernel executing...")
    k2 = K_recursive(t, t, n)
    print("kernel(t, t, n) ", k2)
    print("done")
    print("\nlast kernel executing...")
    res = K_recursive(s, t, n) / sqrt(k1 * k2)
    print("done. returning: ", res)
    return res



def main():
    S = "cells interlinked within cells interlinked"
    T = "within one stem and dreadfully distinct"

    n = 5

    res = K_recursive(S, T, n)

    print(res)



    # control_values = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
    # s = "science is organized knowledge"
    # t = "wisdom is organized life"
    #
    # for n in range(1, 7):
    #     K = K_norm(s, t, n)
    #     print(n, K, 'should be', control_values[n - 1])


if __name__ == '__main__':
    main()
