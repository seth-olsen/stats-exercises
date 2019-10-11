# monte carlo to set bar for test statistic

# create unit rectangular pulse as 1 x arr_size array
# with 0's at every index except for the 
# pulse_len consecutive 1's starting at index
# (pulse_end - pulse_len) and ending at index
# (pulse_end - 1)
# for arr_size < 1, ValueError is raised
# for pulse_end not in [1, arr_size], ValueError is raised
# for pulse_len not in [1, arr_size], ValueError is raised
# for pulse_end < pulse_len, the pulse wraps around the end

import numpy as np
from matplotlib import pyplot as plt
from csv import writer
#from time import time
import sys

def write_data_txt(data, name='data'):
    filename = name
    if filename[-4:] != '.txt':
        filename += '.txt'
    f = open(filename, 'w')
    for entry in data:
        f.write(str(entry) + '\n')
    f.close()
    return

def write_listdata_csv(data, name='data'):
    filename = name
    if filename[-4:] != '.csv':
        filename += '.csv'
    f = open(filename, 'w', newline='')
    wr = writer(f)
    for row in data:
        wr.writerow([str(row)])
    f.close()
    return

def read_zstat_files(start_ind, max_ind, name='mc_zstats', fext='.csv'):
    delim = ','
    if fext != '.csv':
        delim = '\n'
    arrs = []
    for i in range(start_ind, max_ind):
        filename = name + str(i) + fext
        arrs.append(np.genfromtxt(filename, delimiter=delim))
    return np.concatenate(arrs)

def combine_zstat_files(start_ind, max_ind, name='max10e6t100_stats', fext='.csv'):
    oname = name + '_' + str(start_ind) + '_' + str(max_ind) + '.csv'
    np.savetxt(oname, read_zstat_files(start_ind, max_ind, \
                                       name=name, fext=fext), \
               delimiter=',')
    print(oname, ' successfully written') 
    return

def plot_hist(data, figname='mc_hist', bin_edges=[]):
    fig = plt.figure()
    n, bins, patches = plt.hist(data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Max Text Statistic')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Results')
    if figname[-4:] != '.png':
        figname += '.png'
    fig.savefig(figname)
    return

def plot_hist_file(dataname, figname='mc_hist', bin_edges=[]):
    if dataname[-4:] != '.csv':
        dataname += '.csv'
    data = np.genfromtxt(dataname, delimiter=',')
    plot_hist(data, figname=figname, bin_edges=bin_edges)
    return



def rect_pulse(arr_size, pulse_end, pulse_len):
    if arr_size < 1:
        raise ValueError('arr_size < 1')
    if (pulse_end < 1) or (pulse_end > arr_size):
        raise ValueError('pulse_end out of range')
    if (pulse_len < 1) or (pulse_len > arr_size):
        raise ValueError('pulse_len out of range')
    pulse_ind = pulse_end - pulse_len
    pulse_arr = np.zeros(arr_size)
    while pulse_ind < pulse_end:
        pulse_arr[pulse_ind] = 1
        pulse_ind += 1
    return pulse_arr


def complex_wgn(arr_size, mean=0, var=1):
    sd = np.sqrt(var / 2.0)
    re_part = np.random.normal(mean, sd, arr_size)
    im_part = np.random.normal(mean, sd, arr_size)
    return re_part + 1j*im_part


def mc_wgn_zstats(arr_size, pulse_len, ntrials=1):
    trial = 0
    z_stats = np.zeros(ntrials)
    s_hat = rect_pulse(arr_size, arr_size, pulse_len)
    ft_s_hat = np.fft.fft(s_hat)
    #start_time = time()
    while trial < ntrials:
        data = complex_wgn(arr_size)
        ft_data = np.fft.fft(data)
        ft_z = ft_data * ft_s_hat
        z = np.fft.ifft(ft_z)
        z_stats[trial] = max(np.absolute(z))
        trial += 1
    #print('completed in', time() - start_time, 'seconds')
    return z_stats


def main(outname, sample_size=1000000, tau=100, ntrials=100000):
    data = mc_wgn_zstats(sample_size, tau, ntrials=ntrials)
    write_data_csv(data, name=outname)
    return 0

#fname = 'mc_zstats'
#if len(sys.argv) > 1:
#    fname += sys.argv[1]

#main(fname)

#plot_hist('mc_zstats_test.csv', figname='hist-test')

combine_zstat_files(149, 200, name='max10e6t100tt500stats')

combine_zstat_files(149, 200, name='max10e5t100tt500stats')




