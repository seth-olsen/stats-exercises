# analyze look-elsewhere effect from monte carlo test stat data files

import numpy as np
from matplotlib import pyplot as plt

alpha = 10**(-4)
nsamples = int(10/alpha)
n1 = []
lef1 = []
n2 = []
lef2 = []
n8 = []
lef8 = []
n50 = []
lef50 = []
n100 = []
lef100 = []
n500 = []
lef500 = []

kw1 = [{'N' : 1000, 'tau' : 1, 'outname' : 'max1te3_stats.csv'}, \
       {'N' : 5000, 'tau' : 1, 'outname' : 'max1t5e3_stats.csv'}, \
       {'N' : 10000, 'tau' : 1, 'outname' : 'max1te4_stats.csv'}, \
       {'N' : 50000, 'tau' : 1, 'outname' : 'max1t5e4_stats.csv'}, \
       {'N' : 100000, 'tau' : 1, 'outname' : 'max1te5_stats.csv'}, \
       {'N' : 500000, 'tau' : 1, 'outname' : 'max1t5e5_stats.csv'}, \
       {'N' : 1000000, 'tau' : 1, 'outname' : 'max1te6_stats.csv'}]

kw2 = [{'N' : 1000, 'tau' : 2, 'outname' : 'max2te3_stats.csv'}, \
       {'N' : 5000, 'tau' : 2, 'outname' : 'max2t5e3_stats.csv'}, \
       {'N' : 10000, 'tau' : 2, 'outname' : 'max2te4_stats.csv'}, \
       {'N' : 50000, 'tau' : 2, 'outname' : 'max2t5e4_stats.csv'}, \
       {'N' : 100000, 'tau' : 2, 'outname' : 'max2te5_stats.csv'}, \
       {'N' : 500000, 'tau' : 2, 'outname' : 'max2t5e5_stats.csv'}, \
       {'N' : 1000000, 'tau' : 2, 'outname' : 'max2te6_stats.csv'}]

kw8 = [{'N' : 1000, 'tau' : 8, 'outname' : 'max8te3_stats.csv'}, \
       {'N' : 5000, 'tau' : 8, 'outname' : 'max8t5e3_stats.csv'}, \
       {'N' : 10000, 'tau' : 8, 'outname' : 'max8te4_stats.csv'}, \
       {'N' : 50000, 'tau' : 8, 'outname' : 'max8t5e4_stats.csv'}, \
       {'N' : 100000, 'tau' : 8, 'outname' : 'max8te5_stats.csv'}, \
       {'N' : 500000, 'tau' : 8, 'outname' : 'max8t5e5_stats.csv'}, \
       {'N' : 1000000, 'tau' : 8, 'outname' : 'max8te6_stats.csv'}]

kw50 = [{'N' : 1000, 'tau' : 50, 'outname' : 'max10e3t50_stats.csv'}, \
        {'N' : 5000, 'tau' : 50, 'outname' : 'max5x10e3t50_stats.csv'}, \
        {'N' : 10000, 'tau' : 50, 'outname' : 'max10e4t50_stats.csv'}, \
        {'N' : 50000, 'tau' : 50, 'outname' : 'max5x10e4t50_stats.csv'}, \
        {'N' : 100000, 'tau' : 50, 'outname' : 'max10e5t50_stats.csv'}, \
        {'N' : 500000, 'tau' : 50, 'outname' : 'max5x10e5t50_stats.csv'}, \
        {'N' : 1000000, 'tau' : 50, 'outname' : 'max10e6t50_stats.csv'}]

kw100 = [{'N' : 1000, 'tau' : 100, 'outname' : 'max10e3t100_stats.csv'}, \
         {'N' : 5000, 'tau' : 100, 'outname' : 'max5x10e3t100_stats.csv'}, \
         {'N' : 10000, 'tau' : 100, 'outname' : 'max10e4t100_stats.csv'}, \
         {'N' : 50000, 'tau' : 100, 'outname' : 'max5x10e4t100_stats.csv'}, \
         {'N' : 100000, 'tau' : 100, 'outname' : 'max10e5t100_stats.csv'}, \
         {'N' : 500000, 'tau' : 100, 'outname' : 'max5x10e5t100_stats.csv'}, \
         {'N' : 1000000, 'tau' : 100, 'outname' : 'max10e6t100_stats.csv'}]

kw500 = [{'N' : 1000, 'tau' : 500, 'outname' : 'max10e3t500_stats.csv'}, \
         {'N' : 5000, 'tau' : 500, 'outname' : 'max5x10e3t500_stats.csv'}, \
         {'N' : 10000, 'tau' : 500, 'outname' : 'max10e4t500_stats.csv'}, \
         {'N' : 50000, 'tau' : 500, 'outname' : 'max5x10e4t500_stats.csv'}, \
         {'N' : 100000, 'tau' : 500, 'outname' : 'max10e5t500_stats.csv'}, \
         {'N' : 500000, 'tau' : 500, 'outname' : 'max5x10e5t500_stats.csv'}, \
         {'N' : 1000000, 'tau' : 500, 'outname' : 'max10e6t500_stats.csv'}]


for kw in kw1:
    n1.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef1.append(np.exp(-bar / 2.0) / alpha)

for kw in kw2:
    n2.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef2.append(np.exp(-bar / 4.0) / alpha)

for kw in kw8:
    n8.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef8.append(np.exp(-bar / 16.0) / alpha)

for kw in kw50:
    n50.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef50.append(np.exp(-bar / 100.0) / alpha)

for kw in kw100:
    n100.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef100.append(np.exp(-bar / 200.0) / alpha)

for kw in kw500:
    n500.append(kw['N'])
    stats = np.genfromtxt(kw['outname'], delimiter=',')
    bar = np.quantile(stats, 1 - alpha)
    lef500.append(np.exp(-bar / 1000.0) / alpha)

# plt.figure(111)
# plt.loglog(n1, lef1, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 1')
# plt.show(block=False)

# plt.figure(222)
# plt.loglog(n2, lef2, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 2')
# plt.show(block=False)

# plt.figure(888)
# plt.loglog(n8, lef8, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 8')
# plt.show(block=False)

# plt.figure(150)
# plt.loglog(n50, lef50, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 50')
# plt.show(block=False)

# plt.figure(110)
# plt.loglog(n100, lef100, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 100')
# plt.show(block=False)

# plt.figure(500)
# plt.loglog(n500, lef500, marker='o')
# plt.xlabel('N')
# plt.ylabel('LEF(N)')
# plt.title('tau = 500')
# plt.show(block=False)

plt.figure(555)
plt.loglog(n500, lef500, marker='o', color='b')
plt.loglog(n100, lef100, marker='v', color='r')
plt.loglog(n50, lef50, marker='s', color='g')
plt.loglog(n8, lef8, marker='d', color='y')
plt.loglog(n2, lef2, marker='1', color='k')
plt.loglog(n1, lef1, marker='*', color='m')
plt.legend(['tau = 500','tau = 100','tau = 50','tau = 8','tau = 2','tau = 1'])
plt.xlabel('N')
plt.ylabel('LEF(N)')
plt.title('all tau')
plt.show(block=False)


#np.savetxt(outname_stats, max_stats, delimiter=',')



