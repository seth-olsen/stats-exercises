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
from csv import reader, writer, QUOTE_NONNUMERIC
from time import time

#sample_size = 10**6
#tau = 100
#alpha = 1 / float(sample_size**2)
#eta = np.sqrt(tau * np.log(1 / alpha))

def write_data_txt(data, name='data'):
    filename = name + '.txt'
    f = open(filename, 'w')
    for entry in data:
        f.write(str(entry) + '\n')
    f.close()
    return

def write_data_csv(data, name='data'):
    filename = name + '.csv'
    f = open(filename, 'w', newline='')
    wr = writer(f)
    for row in data:
        wr.writerow([str(row)])
    f.close()
    return

def read_data_txt(name):
    filename = name
    if filename[-4:] != '.txt':
        filename += '.txt'
    f = open(filename)
    data = [line.rstrip() for line in f]
    f.close()
    return np.array(map(float, data))

def read_data_csv(name):
    filename = name
    if filename[-4:] != '.csv':
        filename += '.csv'
    f = open(filename, newline='')
    data = reader(f, quoting=QUOTE_NONNUMERIC)
    f.close()
    return np.array(data)

def plot_hist(dataname, figname='histogram', bin_edges=[]):
    data = read_data_csv(dataname)
    fig = plt.figure()
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Max Text Statistic')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Results')
    if figname[-4:] != '.png':
        figname += '.png'
    fig.savefig(figname)
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


def mc_noise_zstats(arr_size, pulse_len, ntrials=0):
    if ntrials == 0:
        ntrials = arr_size
    trial = 0
    z_stats = []
    s_hat = rect_pulse(arr_size, arr_size, pulse_len)
    ft_s_hat = np.fft.fft(s_hat)
    start_time = time()
    while trial < ntrials:
        data = complex_wgn(arr_size)
        ft_data = np.fft.fft(data)
        ft_z = ft_data * ft_s_hat
        z = np.fft.ifft(ft_z)
        z_stats.append(max(np.absolute(z)))
        trial += 1
    print('completed in', time() - start_time, 'seconds')
    return z_stats


def main(name='mc_zstats', sample_size=1000000, tau=100, trial_factor=1):
    ntrials = sample_size * trial_factor
    data = mc_noise_zstats(sample_size, tau, ntrials=ntrials)
    write_data_csv(data, name=name)
    return 0

#main(name='mc_zstats_test', sample_size=1000, tau=10)

print(mc_noise_zstats(10, 2, ntrials=3))
    
    
#plot_hist('mc_zstats_test.csv', figname='hist-test')
