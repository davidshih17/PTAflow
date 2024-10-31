import os, sys
from glob import glob
import json

import numpy as np
import scipy.linalg as spl
import scipy.optimize as spo
import scipy.special as sps
from mpmath import hyp1f2

import time

import matplotlib.pyplot as plt

from pulsar import Likelihood, Pulsar, PTA

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--start', default='0',
                    help='number of events')
parser.add_argument('--end', default='100',
                    help='number of events')
#parser.add_argument('--testpoints', default='/het/p4/dshih/jet_images-deep_learning/ML4GW/flow_posterior/testpoint.npy',
#                    help='test point')
parser.add_argument('--samples', default='/het/p4/dshih/jet_images-deep_learning/ML4GW/flow_posterior/flowsamples.npy',
                    help='test point')
parser.add_argument('--npulsar',default='1')
parser.add_argument('--withRN', default=False, action='store_true')


#parser.add_argument('--dataset_dict',default='dataset_dict.npy')
arguments = parser.parse_args(sys.argv[1:])

print(arguments)

start=int(arguments.start)
end=int(arguments.end)
#testpoints=arguments.testpoints
samples=arguments.samples
npulsar=int(arguments.npulsar)
withRN=arguments.withRN


#if not os.path.isdir(os.path.join('.','12p5yr_stochastic_analysis')):
#    !git clone https://github.com/nanograv/12p5yr_stochastic_analysis.git

# Ordered list of most sensitive pulsars, taken from [arXiv:2009.04496]
bestpsrs = ['J1909-3744', 'J2317+1439', 'J2043+1711', 'J1600-3053', 'J1918-0642',
            'J0613-0200', 'J1944+0907', 'J1744-1134', 'J1910+1256', 'J0030+0451']

# Post-fit average errors taken from [arXiv:2009.04496] (proxy for white noise)
besterrs = {'J1909-3744' : 0.061,
            'J2317+1439' : 0.252, 
            'J2043+1711' : 0.151, 
            'J1600-3053' : 0.245,
            'J1918-0642' : 0.299,
            'J0613-0200' : 0.178,
            'J1944+0907' : 0.365,
            'J1744-1134' : 0.307, 
            'J1910+1256' : 0.187,
            'J0030+0451' : 0.200}

NGdatadir = os.path.join('.','12p5yr_stochastic_analysis', 'data')

bestlocs, bestTOAs = {}, {} # Dictionaries for loaded data

with open(os.path.join(NGdatadir, '12p5yr_median.json')) as RNfile:
    RNdata = json.load(RNfile)

with open(os.path.join(NGdatadir, '12p5yr_maxlike.json')) as RNfile:
    RNdataMLE = json.load(RNfile)

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
    #   first/last case processing, so I'm not going to bother -- Marat 
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

pulsar={}
for psr in bestpsrs:
    pulsar[psr]=Pulsar(bestTOAs[psr],location=bestlocs[psr])
    
#pulsar0=Pulsar(bestTOAs['J1909-3744'],location=bestlocs['J1909-3744'])
#dtoyr = 1/365.24
#DeltaTobs=dtoyr*(max(pulsar0.TOAs)-min(pulsar0.TOAs))
#pta=PTA([pulsar0],f_L=0.01,Delta_f=0.01,Nfreq=1000)
pta=PTA([pulsar[psr] for psr in bestpsrs[:npulsar]],f_L=0.01,Delta_f=0.01,Nfreq=1000)
ll=Likelihood(pta)

all_flow_dict=np.load(samples,allow_pickle=True).item()

all_flow_samples=all_flow_dict['samples'][:,start:end,:]
testpoints=all_flow_dict['residuals']

#parmin=np.array([-18,2,-19,-19,-19,-19,-19,1,1,1,1,1])
#parmax=np.array([-13,9,-13,-13,-13,-13,-13,7,7,7,7,7])
#[log10AGW,gammaGW,log10ARN1,...log10ARN5,gammaRN1,...gammaRN5]

    

all_ll_list=[]
for testpoint,flow_samples in zip(testpoints,all_flow_samples):
    
    t0=time.time()
    testpoint_separated=[]
    for psr in bestpsrs[:npulsar]:
        testpoint_separated.append(testpoint[:(len(bestTOAs[psr])-3)].reshape((1,-1)))
        testpoint=testpoint[(len(bestTOAs[psr])-3):]



    ll_list=[]
    for sample in flow_samples:
        ll_list.append((sample.tolist())+[ll.exactLL(testpoint_separated,Aw=0.1,AGWB=10**sample[0],gamma=sample[1],\
                                        Ar=10**sample[2:2+npulsar],gammar=sample[2+npulsar:2+2*npulsar])])
    t1=time.time()
    print(t1-t0,' seconds elapsed')
    print(t1-t0,' total time')

    all_ll_list.append(ll_list)
    
all_ll_list=np.array(all_ll_list)
print(all_ll_list.shape)
    
np.save(samples+'_likelihoods_start_'+str(start)+'_end_'+str(end)+'.npy',all_ll_list)