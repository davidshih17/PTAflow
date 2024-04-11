from micropta import Pulsar, PTA, Likelihood

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

# Clone and load data from NANOGrav 12.5 year analysis
if not os.path.isdir(os.path.join('.','12p5yr_stochastic_analysis')):
    !git clone https://github.com/nanograv/12p5yr_stochastic_analysis.git

# Ordered list of most sensitive pulsars, taken from [arXiv:2009.04496]
bestpsrs = ['J1909-3744', 'J2317+1439', 'J2043+1711', 'J1600-3053', 'J1918-0642',
            'J0613-0200', 'J1944+0907', 'J1744-1134', 'J1910+1256', 'J0030+0451']

NGdatadir = os.path.join('.','12p5yr_stochastic_analysis', 'tutorials', 'data')

bestlocs, bestTOAs = {}, {} # Dictionaries for loaded data

for psr in bestpsrs:
    # Extract the celestial position (elliptic longitude/lattitude) of pulsar
    parloc = glob(os.path.join(NGdatadir, 'par', psr + '*'))[-1]
    with open(parloc) as pfile:
        loc = np.zeros(2, dtype=np.float64)
        for line in pfile:
            if line.split()[0] == 'LAMBDA': # Ecliptic longitude (degrees)
                loc[0] = float(line.split()[1])*np.pi/180
            if line.split()[0] == 'BETA': # Ecliptic latitude (degrees)
                loc[1] = float(line.split()[1])*np.pi/180

    bestlocs[psr] = loc

    # Extract the first (lowest frequency) TOA reported for each observing run
    #   Taking the average TOA might be better, but it would require special
    #   first/last case processing, so we'll do the simpler thing here
    timloc = glob(os.path.join(NGdatadir, 'tim', psr + '*'))[-1]
    with open(timloc) as tfile:
        tdataread = False
        tdatacurrentobs = ''
        tdataTOAs = []
        for line in tfile:
            if not line.startswith('C'):
                if tdataread:
                    obs, obsTOA = line.split()[0], line.split()[2]
                    if not (obs == tdatacurrentobs):
                        tdataTOAs.append(float(obsTOA))
                        tdatacurrentobs = obs
                else:
                    if line[:6] == 'FORMAT':
                        tdataread = True

    bestTOAs[psr] = np.array(tdataTOAs, dtype=np.float64)



# Create synthetic data for a PTA of 5 NANOGrav pulsars
PulsarTop5 = [Pulsar(bestTOAs[psr], location=bestlocs[psr], name=psr) for psr in bestpsrs[:5]]
PTATop5 = PTA(PulsarTop5)

Npulsar = PTATop5.Npulsar
Nsamples = 50000

AGWBuse = 10**-14
Awuse = 0.1
Aruse = 10**np.random.uniform(-18,-13, size=Npulsar)
gammaruse = np.random.uniform(1,7, size=Npulsar)

CorrData = PTATop5.generate_data(Nsamples=Nsamples, AGWB=AGWBuse, Aw=Awuse, Ar=Aruse, gammar=gammaruse)

ExactCorr = Likelihood(PTATop5).ttcorrmat(Aw=Awuse, AGWB=AGWBuse, Ar=Aruse, gammar=gammaruse)

plt.figure(figsize = (5*Npulsar, 5*Npulsar))
for i in range(Npulsar):
    for j in range(Npulsar):
        plt.subplot(Npulsar, Npulsar, Npulsar*i+j+1)
        plt.xlabel(r'$G$-projected $t$'.format(i+1,j+1))
        plt.ylabel(r'$\langle r_{}(0) r_{}(t) \rangle$'.format(i+1, j+1))
        if i == j:
            plt.title(r'$A_{} =$ {:.1e}, $\gamma_{} = {:.2f}$'.format(i, Aruse[i], i, gammaruse[i]))
        plt.plot(ExactCorr[i][j][0], label='exact')

        corr_mean = np.mean((CorrData[i][:,0].reshape(-1,1) * CorrData[j]), axis=0)
        corr_std = np.std((CorrData[i][:,0].reshape(-1,1) * CorrData[j]), axis=0)/np.sqrt(Nsamples)
        plt.plot(corr_mean, label='data')
        plt.fill_between(range(len(corr_mean)), corr_mean-2*corr_std, corr_mean+2*corr_std, color='C1', alpha=0.2)
        plt.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.suptitle(r'Simple correlations, {} set: $A_\mathrm{{GW}} =$ {:.1e}, {} samples'.format('2 Pulsar', AGWBuse, Nsamples), fontsize=14)
plt.savefig('CorrExample2.jpg', bbox_inches='tight')