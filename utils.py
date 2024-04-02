import numpy as np
import torch
from torch.utils.data import Dataset

# transform to -1,1
#parmin=np.array([-18,2,-19,-19,-19,-19,-19,1,1,1,1,1])
#parmax=np.array([-13,9,-13,-13,-13,-13,-13,7,7,7,7,7])

# these are the priors I used for 10p

#log10AGWBvals=np.random.uniform(low=-18,high=-13,size=1000000)
#gammaGWvals=np.random.uniform(low=1,high=7,size=1000000)

#log10ARNvals=np.random.uniform(low=-19,high=-13,size=(1000000,Np))
#gammaRNvals=np.random.uniform(low=1,high=7,size=(1000000,Np))

parmin=np.array([-18,1,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,1,1,1,1,1,1,1,1,1,1])
parmax=np.array([-13,7,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,7,7,7,7,7,7,7,7,7,7])


def loglin_transform(residuals,thresh):
    residuals[residuals>thresh]=np.log(residuals[residuals>thresh]/thresh)+thresh
    residuals[residuals<-thresh]=-np.log(np.abs(residuals[residuals<-thresh]/thresh))-thresh
    return residuals

class MemmapDataset(Dataset):
    def __init__(self,parameters,residuals):
        super().__init__()

        self.residuals=residuals
        self.parameters=parameters
        self.length = len(residuals)


    def __getitem__(self, idx):
        return torch.from_numpy((-1+2*(self.parameters[idx]-parmin)/(parmax-parmin)).astype('float32')),\
                torch.from_numpy(loglin_transform(self.residuals[idx]*1e7,1000).astype('float32'))


    def __len__(self):
        return self.length
