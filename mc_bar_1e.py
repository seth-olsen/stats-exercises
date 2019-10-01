# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import numba
import sys

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

N = 10**4
tau = 100
alpha = 10**(-4)
nsamples = int(10/alpha)
outname_threshold = 'threshold.txt'
outname_stats = 'h0_stats10e4t1.csv'

for i in range(len(sys.argv) - 1):
    if sys.argv[i] == '-N':
        N = int(sys.argv[i+1])
    elif sys.argv[i] == '-tau':
        tau = int(sys.argv[i+1])
    elif sys.argv[i] == '-alpha':
        alpha = float(sys.argv[i+1])
    elif sys.argv[i] == '-nsamples':
        nsamples = int(sys.argv[i+1])
    elif sys.argv[i] == '-outname_threshold':
        outname_threshold = sys.argv[i+1]

ft_signal = np.fft.fft(np.concatenate([np.zeros(N-tau), np.ones(tau)]))
#max_stats = np.zeros(nsamples)
#integrated_stats = np.zeros(nsamples)
#all_stats = np.zeros((nsamples, N))
for sample in range(nsamples):
    #stat = abs2(np.fft.ifft(ft_signal * \
                            #np.fft.fft(np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N))))
    #max_stats[sample] = max(stat)
    #integrated_stats[sample] = np.sum(stat) / float(N)
    #all_stats[sample] = abs2(np.fft.ifft(ft_signal * \
                                         #np.fft.fft(np.random.normal(0, 1, N) + \
                                                    #1j*np.random.normal(0, 1, N))))

#np.savetxt(outname_stats, all_stats, delimiter=',')

np.savetxt(outname_stats, all_stats, delimiter=',')

#threshold = np.quantile(max_stats, 1 - alpha)

#print('\nthe threshold for the non-normalized test statistic is ', threshold)

#print('\n\nthe threshold for the normalized test statistic is ', threshold/float(tau))

#f = open(outname_threshold, 'w')
#f.write('N  =  ' + str(N) + '\n\n')
#f.write('tau  =  ' + str(tau) + '\n\n')
#f.write('alpha  =  ' + str(alpha) + '\n\n')
#f.write('nsamples  =  ' + str(nsamples) + '\n\n')
#f.write('threshold  =  ' + str(threshold) + '\n\n')
#f.write('normalized  =  ' + str(threshold/float(tau)) + '\n\n')
#f.close()

