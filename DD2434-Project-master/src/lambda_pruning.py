from math import sqrt

lam = .5


def Kdoubleprime_LP(s, t, n, m):
    if (min(len(s), len(t))) < n:
        return 0

    x = s[-1]
    u = t[-1]

    if (x == u):
        return lam * (Kdoubleprime_LP(s, t[0:-1], n, m - 1) +
                      lam * Kprime_LP(s[0:-1], t[0:-1], n - 1, m - 2))
    else:
        #	assumes u is just a letter, not a sequence
        return lam * Kdoubleprime_LP(s, t[0:-1], n, m - 1)


def Kprime_LP(s, t, n, m):
    if n == 0:
        return 1

    if min(len(s), len(t)) < n:
        return 0

    if m < 2 * n:
        return 0

    return lam * Kprime_LP(s[0:-1], t, n, m - 1) + Kdoubleprime_LP(s, t, n, m)


def K_LP(s, t, n, m):
    if min(len(s), len(t)) < n:
        return 0

    x = s[-1]
    sum_ = 0
    for j, tj in enumerate(t):
        if tj == x:
            sum_ += lam ** 2 * Kprime_LP(s[0:-1], t[0: j], n - 1, m - 2)

    return K_LP(s[0:-1], t, n, m) + sum_


def K_norm_LP(s, t, n, m):
    k1 = K_LP(s, s, n, m)
    k2 = K_LP(t, t, n, m)
    res = K_LP(s, t, n, m) / sqrt(k1 * k2)
    return res


def test():
    """ Examples to check that it's working """
    print('K_LP(car, car, 1, 100) = ', K_LP('car', 'car', 1, 100), 'should be 3*lambda^2 = .75')
    print('K_LP(car, car, 2, 100) = ', K_LP('car', 'car', 2, 100), ' should be lambda^6 + 2*lambda^4 = 0.140625')
    print('K_LP(car, car, 3, 100) = ', K_LP('car', 'car', 3, 100), 'should be lambda^6 = 0.0156')

    print('K_norm_LP(cat, car, 1, 100) = ', K_norm_LP('cat', 'car', 1, 100), 'should be 2/3')
    print('K_LP(cat, car, 2, 100) = ', K_LP('cat', 'car', 2, 100), 'should be lambda^4 = 0.0625')
    print('K_norm_LP(cat, car, 2, 100) = ', K_norm_LP('cat', 'car', 2, 100), 'should be 1/(2+lambda^2) = 0.44444')

    print(K_LP("AxxxxxxxxxB", "AyB", 2, 100), 'should be =0.5^14 = 0.00006103515625')
    print(K_LP("AxxxxxxxxxB", "AxxxxxxxxxB", 2, 100), 'should be 12.761724710464478')

    print(K_LP("ab", "axb", 2, 100), 'should be =0.5^5 = 0.03125')
    print(K_LP("ab", "abb", 2, 100), 'should be 0.5^5 + 0.5^4 = 0.09375')
    print(K_norm_LP("ab", "ab", 2, 100), 'should be 1')
    print(K_norm_LP("AxxxxxxxxxB", "AxxxxxxxxxB", 2, 100), 'should be 1')

    kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
    for x in range(1, 7):
        print(x,
              K_norm_LP("science is organized knowledge",
                        "wisdom is organized life", x, 100), 'should be',
              kss[x - 1])

    # Without lambda pruning, one common subsequence, "AB" would be found in the following two strings. (With k=2)
    #	SSK_LP(2,"ab","axb")=0.5^14 = 0,00006103515625
    # lambda pruning allows for the control of the match length. So, if m (the maximum lambda exponent) is e.g. 8, these two strings would yield a kernel value of 0:
    # with lambda pruning:    SSK-LP(2,8,"AxxxxxxxxxB","AyB")= 0
    # without lambda pruning: SSK_LP(2,"AxxxxxxxxxB","AyB")= 0.5^14 = 0,00006103515625
    print(K_LP("AxxxxxxxxxB", "AyB", 2, 8), 'should be 0')

    print(K_LP("AxxB", "AyyB", 2, 8), 'should be 0.5^8 = 0,00390625')
    kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
    for x in range(1, 7):
        # Lambda pruning is parametrized by the maximum lambda exponent. It is a good idea to choose that value to be about 3 or 4 times the subsequence length as a rule of thumb
        m = x * 4
        print(x,
              K_norm_LP("science is organized knowledge",
                        "wisdom is organized life", x, m))


if __name__ == "__main__":
    test()
