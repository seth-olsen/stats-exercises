# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import sys

def abs2(x):
    return x.real**2 + x.imag**2

def get_signal(arr_size, sig_width, start=0):
    over = arr_size - sig_width - start
    if over < 0:
        return np.concatenate((np.ones(-over), np.zeros(arr_size - sig_width), \
                               np.ones(sig_width + over)))
    else:
        return np.concatenate((np.zeros(start), np.ones(sig_width), np.zeros(over)))

def get_tri_right(arr_size, tri_width, start=0):
    tri_r = np.zeros(arr_size)
    for j in np.arange(start, start + tri_width):
        tri_r[j] = tri_width - j
    return tri_r / np.linalg.norm(tri_r)

def get_wgn(arr_size):
    return np.random.normal(0,1,arr_size) + 1j*np.random.normal(0,1,arr_size)
    
# minimum and maximum amplitudes in range of plotting Power(Amplitude)
Amax = 4.0
Amin = 0.05
bar5 = 13924.8
bar6 = 15317.4

N = 10**6 # size of dataset
tau = 100 # width of probe (template)
tautri = 500 # width of convolution filter
alpha = 10**(-4) # desired overall false alarm probability
Nmc = 10000 # number of sample datasets to draw via monte carlo 
threshold = bar6
outname = 'pwrA2_6.csv'

# take user-input parameters from keyworded command line arguments
# input arg_name=arg_val as: -<arg_name> <arg_val>
for i in range(len(sys.argv) - 1):
    if sys.argv[i] == '-N':
        N = int(sys.argv[i+1])
    elif sys.argv[i] == '-tau':
        tau = int(sys.argv[i+1])
    elif sys.argv[i] == '-tautri':
        tautri = int(sys.argv[i+1])
    elif sys.argv[i] == '-alpha':
        alpha = float(sys.argv[i+1])
    elif sys.argv[i] == '-Nmc':
        Nmc = int(sys.argv[i+1])
    elif sys.argv[i] == '-outname':
        outname = sys.argv[i+1]
    elif sys.argv[i] == '-Amin':
        Amin = float(sys.argv[i+1])
    elif sys.argv[i] == '-Amax':
        Amax = float(sys.argv[i+1])
    elif sys.argv[i] == '-normd_threshold':
        threshold = float(sys.argv[i+1])

if outname[-4:] != '.csv':
    outname += '.csv'

# set monte carlo parameters
Avals = np.arange(Amin, Amax + Amin, Amin)
namps = np.size(Avals)
counts = np.zeros(namps)
# construct signal template for z0 convolution
srev_ft = np.fft.fft(np.concatenate((np.ones(1),np.zeros(N-tau),np.ones(tau-1))))
tt_ft = np.fft.fft(get_tri_right(N, tautri))
tt_ft_cc = np.conj(tt_ft)
# get z_inc(n_s) = Amin \hat{s}(\tau, n_s) M(\tau_{tri}) \hat{s}(\tau_I, n_sI)
nsI = int(N / 2) # and letting tau_I = tau, phi_I = 0
zinc = Amin * np.fft.ifft(srev_ft * np.fft.fft(get_signal(N,tau,start=nsI)) \
                          / abs2(tt_ft))
# now iterate over noise realizations
for i in range(Nmc):
    # compute pure noise component of z
    z0 = np.fft.ifft(srev_ft * np.fft.fft(get_wgn(N)) / tt_ft_cc)
    Aind = 0
    # compute bar array for efficient testing
    bar_im_z0sq = threshold - (z0.imag)**2
    # compute real part of z with with Amin
    re_z = z0.real + zinc
    while (np.sum(re_z**2 > bar_im_z0sq) == 0) and (Aind < namps):
        # increment real part of z until first amp that passes bar
        Aind += 1
        re_z += zinc
    # increment counts of all amps that pass bar
    counts += np.concatenate([np.zeros(Aind), np.ones(namps - Aind)])

# add A = 0 and Pi(A=0) = alpha to arrays, and make counts a probability array    
counts = np.concatenate([np.full(1, alpha), counts / float(Nmc)])
Avals = np.concatenate([np.zeros(1), Avals])
# save Pi(A) data from discretized interval
np.savetxt(outname, counts, delimiter=',')
amp_outname = 'amps_' + outname
np.savetxt(amp_outname, Avals, delimiter=',')
