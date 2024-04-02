import numpy as np
import os
import torch
import torch.nn as nn

from utils import MemmapDataset
from models import EmbeddingLayer
from models import define_model

from tqdm import tqdm

def train(model,optimizer,train_loader,noise_data=0,noise_context=0):
    
    model.train()
    train_loss = 0

#    pbar = tqdm(total=len(train_loader.dataset),leave=True)
    for batch_idx, (data,cond_data) in enumerate(tqdm(train_loader)):
        data+=noise_data*torch.normal(mean=torch.zeros(data.shape),std=torch.ones(data.shape))
        data = data.to(device)

        cond_data = cond_data.float()
        cond_data+=noise_context*torch.normal(mean=torch.zeros(cond_data.shape),std=torch.ones(cond_data.shape))
        cond_data = cond_data.to(device)
        
        optimizer.zero_grad()
        # print(data, cond_data)
        loss = -model(data, cond_data).mean()
        loss.backward(retain_graph=True)

        train_loss += float(loss.item())

    
#        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        
        optimizer.step()
#        pbar.update(data.size(0))
#        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
#            train_loss / (batch_idx + 1)))


# validation

def val(model,val_loader):

    valloss=0
    model.eval()
    with torch.no_grad():

        for batch_idx, (data,cond_data) in enumerate(val_loader):
            data = data.to(device)

            cond_data = cond_data.float()
            cond_data = cond_data.to(device)

            valloss+=-model(data, cond_data).sum()

    valloss=valloss.cpu().detach().numpy()
    valloss=valloss/len(val_loader.dataset)
    
    return valloss


os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'


CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}



babydataset_train = np.memmap('../flow_posterior/babydataset_residuals_1M_fLDeltaf0.01_10p_withRN_v2.npy_train.memmap', dtype='float32', mode='r', shape=(900000, 2940))
parameters_train=np.load('../flow_posterior/babydataset_parameters_1M_fLDeltaf0.01_10p_withRN_v2.npy_train.npy')


babydataset_val = np.memmap('../flow_posterior/babydataset_residuals_1M_fLDeltaf0.01_10p_withRN_v2.npy_val.memmap', dtype='float32', mode='r', shape=(100000, 2940))
parameters_val=np.load('../flow_posterior/babydataset_parameters_1M_fLDeltaf0.01_10p_withRN_v2.npy_val.npy')

# this comes from the following:
'''
Ordered list of most sensitive pulsars, taken from [arXiv:2009.04496]
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

NGdatadir = os.path.join('/home/shih/work/ML4GW/flow_posterior/','12p5yr_stochastic_analysis', 'data')

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

    ntoalist=[len(bestTOAs[psr])-3 for psr in bestpsrs[:10]]
'''


ntoalist=[405, 444, 299, 233, 259, 275, 133, 265, 167, 460]


print(parameters_train.shape,babydataset_train.shape)
print(parameters_val.shape,babydataset_val.shape)

traindataset = MemmapDataset(parameters_train,babydataset_train)
valdataset = MemmapDataset(parameters_val,babydataset_val)



embedding=EmbeddingLayer(ntoalist,noutput=50)

model=define_model(nfeatures=22,nhidden=2,hidden_size=200,embedding=embedding,dropout=0,nembedding=50)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    embedding = nn.DataParallel(embedding)
    model = nn.DataParallel(model)
embedding=embedding.to(device)
model=model.to(device)


# Use the standard pytorch DataLoader
batch_size = 256*int(torch.cuda.device_count())
print('batch size: ',batch_size)
#batch_size = 256
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(valdataset, batch_size=test_batch_size, shuffle=False)
#,num_workers=4, pin_memory=True)

# train
optimizer = torch.optim.RAdam(model.parameters()) #,lr=1e-4)#, lr=1e-4)

for epoch in range(100):
    print('\n Epoch: {}'.format(epoch))
    train(model,optimizer,trainloader,noise_data=0.,noise_context=0.)
    valloss=val(model,valloader)
    print('epoch '+str(epoch)+' val loss: ',valloss)
    torch.save(model.state_dict(),"babydataset_fLDeltaf0.01_epoch_"+str(epoch)+"_Model.par")
    torch.save(optimizer.state_dict(),"babydataset_fLDeltaf0.01_epoch_"+str(epoch)+"_Optim.par")
