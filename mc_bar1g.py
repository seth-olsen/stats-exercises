# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import numba
import sys
from matplotlib import pyplot as plt

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

N = 10**5
tau_start = 2
tau_end = 500
alpha = 10**(-4)
nsamples = 2#int(10/alpha)
outname_threshold = 'threshold.txt'
outname_stats = 'stats10e6t2_500.csv'

for i in range(len(sys.argv) - 1):
    if sys.argv[i] == '-N':
        N = int(sys.argv[i+1])
    elif sys.argv[i] == '-tau_start':
        tau_start = int(sys.argv[i+1])
    elif sys.argv[i] == '-tau_end':
        tau_end = int(sys.argv[i+1])
    elif sys.argv[i] == '-alpha':
        alpha = float(sys.argv[i+1])
    elif sys.argv[i] == '-nsamples':
        nsamples = int(sys.argv[i+1])
    elif sys.argv[i] == '-outname_stats':
        outname_stats = sys.argv[i+1]
    elif sys.argv[i] == '-outname_threshold':
        outname_threshold = sys.argv[i+1]

taus = np.arange(tau_start, tau_end + 1)
Nt = np.size(taus)
ft_signals = np.empty((Nt, N), dtype=np.csingle)
for j, tt in enumerate(taus):
    ft_signals[j] = np.fft.fft(np.concatenate([np.zeros(N-tt), np.ones(tt)]))
max_tau_stats = np.empty(nsamples)
tau_stats = np.empty(Nt)
#all_stats = np.zeros((nsamples, N))
for i in np.arange(nsamples):
    ft_noise = np.fft.fft(np.random.normal(0, 1, N) + \
                          1j*np.random.normal(0, 1, N))
    for j in np.arange(Nt):
        tau_stats[j] = max(abs2(np.fft.ifft(ft_signals[j] * ft_noise))) / float(taus[j])
    #plt.figure(987)
    plt.plot(taus, tau_stats)
    #plt.show()
    max_tau_stats[i] = max(tau_stats)
    #all_stats[sample] = abs2(np.fft.ifft(ft_signal * \
                                         #np.fft.fft(np.random.normal(0, 1, N) + \
                                                    #1j*np.random.normal(0, 1, N))))
plt.show(block=False)
np.savetxt(outname_stats, max_tau_stats, delimiter=',')

#threshold = np.quantile(max_stats, 1 - alpha)

#f = open(outname_threshold, 'w')
#f.write('N  =  ' + str(N) + '\n\n')
#f.write('tau  =  ' + str(tau) + '\n\n')
#f.write('alpha  =  ' + str(alpha) + '\n\n')
#f.write('nsamples  =  ' + str(nsamples) + '\n\n')
#f.write('threshold  =  ' + str(threshold) + '\n\n')
#f.close()

