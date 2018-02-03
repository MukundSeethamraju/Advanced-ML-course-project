from SSK_recursive import K_recursive
from SSK import kernel as K_DP
from time import process_time
import numpy as np
import matplotlib.pyplot as plt

def plot_string_lengths(n = 3, chunksize = 10, iterations = 5):
    """plt time (x-axis) to run kernel for different length of strings (y-axis)"""
    max = 10
    times = np.zeros((2, max))
    lengths = np.zeros(max)
    errors = np.zeros((2, max))

    for i in range(max):
        threshold = chunksize*i
        lengths[i] = threshold
        s = S[:threshold]

        step = np.zeros((2, iterations))

        for t in range(iterations):
            start = process_time()
            K_recursive(s, s, n)
            stop = process_time()
            time_rc = stop - start
            step[0][t] = time_rc

            start = process_time()
            K_DP(s, s, n)
            stop = process_time()
            time_dp = stop - start
            step[1][t] = time_dp

        errors[0][i] = np.std(step[0])
        errors[1][i] = np.std(step[1])
        times[0][i] = np.mean(step[0])
        times[1][i] = np.mean(step[1])

    times = np.log(times)
    lengths = np.log(lengths)
    plt.plot(lengths, times[0], label='Recursive')
    plt.plot(lengths, times[1], label='DP')
    plt.legend(loc='upper left')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Log(Process time)')
    plt.xlabel('Length of Evaluated Strings (' + str(chunksize) + '-' + str(chunksize*10) + ')')
    plt.show()


def plot_n_lengths(n = 3, limit_string_size = 50, iterations = 5):
    """plt time (x-axis) to run kernel for different subsequence length n (y-axis)"""
    s = S[:limit_string_size]
    times = np.zeros((2, n))
    ns = np.arange(n)
    errors = np.zeros((2, n))

    for n in range(1, n):
        step = np.zeros((2, iterations))

        for t in range(iterations):
            start = process_time()
            K_recursive(s, s, n)
            stop = process_time()
            time_rc = stop - start

            step[0][t] = time_rc

            start = process_time()
            K_DP(s, s, n)
            stop = process_time()
            time_dp = stop - start

            step[1][t] = time_dp

        errors[0][n] = np.std(step[0])
        errors[1][n] = np.std(step[1])
        times[0][n] = np.mean(step[0])
        times[1][n] = np.mean(step[1])

    ns = np.log(ns)
    times = np.log(times)
    plt.plot(ns, times[0], label='Recursive')
    plt.plot(ns, times[1], label='DP')
    plt.legend(loc='upper left')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Log(Process time)')
    plt.xlabel('Substring Length n (1-' + str(n) + ')')

    plt.show()


S = "united states dependency foreign oil sources may reach record levels mids according john h lichtblau president petroleum industry research associates lichtblau speaking alternative energy conference said us may depend foreign suppliers much pct oil surpasssing previous high level pct long term growth dependency foreign oil inevitable lichtblau said much pct us oil imports could come opec nations said lichtblau said us depended foreign suppliers pct oil predicted would increase pct however rate growth affected positively negatively government action inaction lichtblau said said one governments negative actions maintenance windfall profits tax acts disincentive developing existing fields reduces cash flow oil exploration lichtblau called adoption international floor price crude oil help stabilize world oil prices international floor price adopted industrial countries would clearly much effective measure would much less distortive us imposed alone lichtblau said development alternate energy sources synthetic fuels well increased development alaska could lessen us dependency foreign oil lichtblau said potential alternative supplies could limit willingness opec nations raise oil prices said lichtblau also called federal government offer tax abatements oil drilling fill strategic petroleum reserve faster rate develop pilot plans alternative energy reuter"

def main():

    # plot different n
    # plot_n_lengths(n=7, limit_string_size=40, iterations=5)

    # plot different string lengths
    plot_string_lengths(n=2, chunksize=30, iterations=5)


if __name__ == '__main__':
    main()