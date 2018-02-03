from Gram_matrix import GramCalc
from approximation import gram_matrix_approx, form_S
from SSK import kernel
from time import process_time
import numpy as np
import matplotlib.pyplot as plt


# from approximation import gram_matrix_approx, form_S

a = 'iranian foreign minister ali akbar velayati twoday official visit informed cuban foreign ministry officials monday tense situation gulf diplomatic sources said said envoys trip followed tuesday visit nicaragua could linked possible mediation nonaligned movement sevenyearold iraniraq '
b = 'fortune systems corp said shareholders approved sale computer hardware business sci technologies inc transaction expected close week annual meeting fortune said shareholders also voted change fortunes name tigera inc principal subsidiary tigeral corp reuter'
c = 'southmark corp said acquired longterm care facilities containing approximately mln dlrs cash said facilities contain approximately beds seven western states bought bybee associates salemore acquistion brings health care facilities acquired last three months company said reuter'
d = 'raytech corp said acquired raybestos industrieprodukte gmbh mln dlrs raybestos manufacturing facilities radevormwald west germany produces friction materials use clutch braking applications reuter'
e = 'bank england said provided money market mln stg help morning session compares banks estimate system would face shortage around mln stg today central bank bought bank bills outright comprising two mln stg band two pct mln stg band three pct mln stg band three pct reuter'
f = 'unocal corp said raised posted prices us grades crude oil cts barrel effective october move brings price company pay us benchmark grade west texas intermediate west texas sour dlrs barrel price last changed september unocal said reuter'
g = 'kg saur germanbased publisher databases legal bilbiographic reference material said sold assets butterworth group division reed international plc reedl mln dlrs saur said klaus saur president owner company remain president saur operations munich london new york reuter'
h = 'canadian imperial oil pct exxon owned said raised posting light sweet crude oil edmonton canadian cts barrel effective today company said new posting light sweet crude oil edmonton canadian dlrs barrel reuter'
i = 'permian corp subsidiary national intergroup said raised crude oil postings cts barrel effective june company said new posted price west texas intermediate west texas sour dlrs barrel light louisiana sweet price hike follows increases industrywide reuter'

lim = 150
#
all_docs = [a[:lim], b[:lim], c[:lim], d[:lim], e[:lim], f[:lim], g[:lim], h[:lim], i[:lim]]

# all_docs = [a,b,c,d,e,f,g,h,i]


def plot_ns(n=2, iterations=5):
    size = len(all_docs)

    num_of_features = 65

    times = np.zeros((2, size))
    points = np.arange(size)
    errors = np.zeros((2, size))

    for i in range(0, size):
        docs = all_docs[:i + 1]

        step = np.zeros((2, iterations))
        for t in range(iterations):

            start = process_time()
            GC_train = GramCalc(docs, docs, n, kernel=kernel, symmetric=True)
            GC_train.calculate(parallel=False)
            stop = process_time()
            time_original = stop - start
            step[0][t] = time_original

            start = process_time()
            set_of_ngrams = form_S(docs, n, num_of_features)
            gram_matrix_approx(n, set_of_ngrams, docs, docs)
            stop = process_time()
            time_approx = stop - start
            step[1][t] = time_approx

        errors[0][i] = np.std(step[0])
        errors[1][i] = np.std(step[1])
        times[0][i] = np.mean(step[0])
        times[1][i] = np.mean(step[1])

    plt.errorbar(points, times[0], errors[0], linestyle='None', marker='.', label = 'SSK')
    plt.errorbar(points, times[1], errors[1], linestyle='None', marker='.', label = 'Approximating Kernel')
    # plt.plot(lengths.T, times.T)
    plt.legend(loc='upper left')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Log(Process time)')
    plt.xlabel('Number of Documents (1-' + str(size) + ')')
    plt.show()


plot_ns()
