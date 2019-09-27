# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import numba
import sys

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

N = 10**6
tau = 100
alpha = 10**(-4)
nsamples = int(100/alpha)
outname = 'threshold.txt'

for i in range(len(sys.argv) - 1):
    if sys.argv[i] == '-N':
        N = int(sys.argv[i+1])
    elif sys.argv[i] == '-tau':
        tau = int(sys.argv[i+1])
    elif sys.argv[i] == '-alpha':
        alpha = float(sys.argv[i+1])
    elif sys.argv[i] == '-nsamples':
        nsamples = int(sys.argv[i+1])
    elif sys.argv[i] == '-outname':
        outname = sys.argv[i+1]

ft_signal = np.fft.fft(np.concatenate([np.zeros(N-tau), np.ones(tau)]))
max_stats = np.zeros(nsamples)
for sample in range(nsamples):
    ft_noise = np.fft.fft(np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N))
    max_stats[sample] = max(abs2(np.fft.ifft(ft_noise * ft_signal)))

threshold = np.quantile(max_stats, 1 - alpha)

print('\nthe threshold for the non-normalized test statistic is ', threshold)

print('\n\nthe threshold for the normalized test statistic is ', 2*threshold/float(tau))

f = open(outname, 'w')
f.write('N  =  ' + str(N) + '\n\n')
f.write('tau  =  ' + str(tau) + '\n\n')
f.write('alpha  =  ' + str(alpha) + '\n\n')
f.write('nsamples  =  ' + str(nsamples) + '\n\n')
f.write('threshold  =  ' + str(threshold) + '\n\n')
f.write('normalized  =  ' + str(2*threshold/float(tau)) + '\n\n')
f.close()

