from micropta import Pulsar, PTA, Likelihood

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

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

PulsarTop5 = [Pulsar(bestTOAs[psr], location=bestlocs[psr], name=psr) for psr in bestpsrs[:5]]
PTATop5 = PTA(PulsarTop5)

Npulsar = PTATop5.Npulsar
Nsamples = 250

AGWBuse = 10**np.linspace(-16,-11,21)
Awuse = 0.1
Aruse = np.full(Npulsar, 10**-14)
gammaruse = np.full(Npulsar, 2.5)

OSAvg = []
OSDist = []
for AGW in AGWBuse:
    OSData = PTATop5.generate_data(Nsamples=Nsamples, AGWB=AGW, Aw=Awuse, Ar=Aruse, gammar=gammaruse)
    OSResult = Likelihood(PTATop5).optimalstatistic(OSData, Aw=Awuse, Ar=Aruse, gammar=gammaruse)
    OSDist.append(OSResult)
    OSAvg.append(np.mean( np.sqrt(np.abs(OSResult)) ))

plt.loglog(AGWBuse, OSAvg)
plt.loglog(AGWBuse, AGWBuse, 'k--')
plt.xlabel('Injected $A_\mathrm{{GW}}$', fontsize=14)
plt.ylabel(r'Optimal statistic $|\hat{A}|$', fontsize=14)

plt.savefig('OSExample.jpg', bbox_inches='tight')