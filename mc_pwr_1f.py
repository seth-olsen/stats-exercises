# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import numba
import sys
from csv import writer

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

Amax = 10.0
Amin = 0.05

N = 10**6
tau = 100
alpha = 10**(-4)
nsamples = 1000
threshold = 30.0
outname = 'power.csv'

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
    elif sys.argv[i] == '-Amin':
        Amin = float(sys.argv[i+1])
    elif sys.argv[i] == '-Amax':
        Amax = float(sys.argv[i+1])
    elif sys.argv[i] == '-threshold':
        threshold = float(sys.argv[i+1])

#pulse = lambda ns: np.concatenate([np.zeros(ns), np.ones(tau), np.zeros(N-tau-ns)])
#eiphi = lambda phis: np.cos(phis) + 1j*np.sin(phis)
#phase = eiphi(2 * np.pi * np.random.random(nsamples))
#n_s = np.random.randint(1, N - tau - 1, nsamples)

Avals = np.arange(Amin, Amax + Amin, Amin)
pwrs = []

ft_signal = np.fft.fft(np.concatenate([np.zeros(N-tau), np.ones(tau)]))

# for A in Avals:
#     count = 0
#     for i in range(nsamples):
#         ft_data = np.fft.fft(np.random.normal(0, 1, N) + \
#                              1j*np.random.normal(0, 1, N) + \
#                              A*phase[i]*pulse(n_s[i]))
#         if max(abs2(np.fft.ifft(ft_data * ft_signal))) > threshold:
#             count += 1
#     pwrs.append(count / float(nsamples))

for A in Avals:
    count = 0
    bar = tau*threshold - (tau*A)**2
    for i in range(nsamples):
        ft_noise = np.fft.fft(np.random.normal(0, 1, N) + \
                              1j*np.random.normal(0, 1, N))
        z0 = np.fft.ifft(ft_noise * ft_signal)
        if max(2*tau*A*z0.real + abs2(z0)) > bar:
            count += 1
    pwrs.append(count / float(nsamples))


f = open(outname, 'w', newline='')
wr = writer(f)
wr.writerow([str(alpha)])
for val in pwrs:
    wr.writerow([str(val)])
f.close()

outname_amp = 'amps_' + outname
f_amp = open(outname_amp, 'w', newline='')
wr_amp = writer(f_amp)
wr_amp.writerow(['0.0'])
for val in Avals:
    wr_amp.writerow([str(val)])
f_amp.close()



