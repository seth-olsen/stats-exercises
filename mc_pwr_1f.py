# monte carlo to set bar for test statistic in problem 1e

import numpy as np
import numba
import sys
from csv import writer
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# for modulus squared of complex array
#@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
#def abs2(x):
#    return x.real**2 + x.imag**2

# minimum and maximum amplitudes in range of plotting Power(Amplitude)
Amax = 5.0
Amin = 0.05
# the power for which you want to find the minimum amplitude needed
desired_pwr = 0.5

N = 10**6 # size of dataset
tau = 100 # width of probe (template)
alpha = 10**(-4) # desired overall false alarm probability
nsamples = 1000 # number of sample datasets to draw via monte carlo 
normd_threshold = 10.0 # threshold set by monte carlo and normalized to width
outname = 'power_test.csv' # output file name


# take user-input parameters from keyworded command line arguments
# input arg_name=arg_val as: -<arg_name> <arg_val>
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
    elif sys.argv[i] == '-desired_pwr':
        desired_pwr = float(sys.argv[i+1])
    elif sys.argv[i] == '-normd_threshold':
        normd_threshold = float(sys.argv[i+1])

# generate unit height rectangular pulse of length tt starting at index ns  
#pulse = lambda ns, tt: np.concatenate([np.zeros(ns), np.ones(tt), np.zeros(N-tt-ns)])
# generate array of complex phases from array of angles (in radians)
#eiphi = lambda phis: np.cos(phis) + 1j*np.sin(phis)

# get array of amplitude values to build Power(Amplitude)
Avals = np.arange(Amin, Amax + Amin, Amin)
# get detection bar for non-normalized test statistic
bar = tau*normd_threshold
# get fourier transform of probe signal (will be multiplied by noise element-wise
# before being intverted back as time domain test statistic z_0 under null H0)
ft_signal = np.fft.fft(np.concatenate([np.zeros(N-tau), np.ones(tau)]))

# ***compute test statistic w/setting tau_I=tau and randomizing phi_I, n_sI***

# get random values of n_s, phi for each monte carlo sample
#phase = eiphi(2 * np.pi * np.random.random(nsamples))
#n_s = np.random.randint(1, N - tau - 1, nsamples)
# INEFFICIENT
# pwrs = []
# for A in Avals:
#     count = 0
#     for i in range(nsamples):
#         ft_data = np.fft.fft(np.random.normal(0, 1, N) + \
#                              1j*np.random.normal(0, 1, N) + \
#                              A*phase[i]*pulse(n_s[i]))
#         if max(abs2(np.fft.ifft(ft_data * ft_signal))) > bar:
#             count += 1
#     pwrs.append(count / float(nsamples))

# ***compute test statistic w/setting phi_I=0, n_sI=tau, tau_I=tau***

# get H1(injection) piece of test statistic from unit injection
# (to be multiplied by amplitude to get injected contribution z_I to z = z_0 + z_I)
#zI = np.fft.ifft(ft_signal * np.fft.fft(pulse(tau, tau)))
# know that this will be real triangular pulse, so more efficient and precise way:
zI = np.zeros(N)
zI[tau] = float(tau)
for pos in range(1, tau):
    zI[pos] = float(pos)
    zI[2*tau - pos] = float(pos)
# and for computational purposes can just use incremental array (explained later)
zI = Amin*zI

namps = np.size(Avals) # total number of amplitudes to check
counts = np.zeros(namps) # array counting how many bars passed for each amplitude
# compute min. amp. needed to pass bar for (nsamples) monte carlo samples of WGN
for i in range(nsamples):
    # compute noise contribution to z 
    z0 = np.fft.ifft(ft_signal * \
                     np.fft.fft(np.random.normal(0, 1, N) + \
                                1j*np.random.normal(0, 1, N)))
    Aind = 0
    # find smallest amplitude that passes bar
    #while (max(abs2(Avals[Aind]*zI + z0)) < bar) and (Aind < namps):
    #    Aind += 1
    
    # more efficient way after setting phi_I=0:
    #re_z0 = z0.real
    #im_z0sq = (z0.imag)**2
    #while (max((Avals[Aind]*zI + re_z0)**2 + im_z0sq) < bar) and (Aind < namps):
    #    Aind += 1

    # even more efficient way since the only thing changing is linear increment of z.real:
    # get array of bars set by part of test that doesn't depend on amplitude
    bar_im_z0sq = bar - (z0.imag)**2
    # get real part of z that will be incremented by zI = Amin*zI(unit)
    re_z = z0.real + zI
    # now keep incrementing as long as no element of re_z(A)**2 > bar_im_z0sq is True 
    while (np.sum(re_z**2 > bar_im_z0sq) == 0) and (Aind < namps):
        Aind += 1
        re_z += zI
    # once condition re_z**2 > im_z0sq_bar is met by at least one element of re_z(A)**2,
    # increment the counts for all amplitudes >= A
    counts += np.concatenate([np.zeros(Aind), np.ones(namps - Aind)])

# turn counts into probability array and prepend false alarm Power(A=0) = alpha
counts = np.concatenate([np.full(1, alpha), counts / float(nsamples)])
Avals = np.concatenate([np.zeros(1), Avals])
# interpolate function Power(A) shifted down by desired_pwr =>
# the zero of this function gives the min. amp. needed for desired_pwr
amp_for_desired_pwr = spline(Avals, counts - desired_pwr).roots()
print('\nfor power = ', desired_pwr, '\nneed amplitude = ', amp_for_desired_pwr, '\n')

# write pwr and amp files in case in case you want to plot pwr(A) or redo spline
f = open(outname, 'w', newline='')
wr = writer(f)
for val in counts:
    wr.writerow([str(val)])
f.close()

outname_amp = 'amps_' + outname
f_amp = open(outname_amp, 'w', newline='')
wr_amp = writer(f_amp)
for val in Avals:
    wr_amp.writerow([str(val)])
f_amp.close()



