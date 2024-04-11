from micropta import Pulsar, PTA, Likelihood

import numpy as np
import matplotlib.pyplot as plt

regularTOAs = 53216 + 14. * np.arange(337)
SimplePTA = PTA([
                    Pulsar(regularTOAs, name='PulsarA'),
                    Pulsar(regularTOAs, name='PulsarB'),
                ], covmat = np.array([[1.0, 0.3], [0.3, 1.0]])
                )

Npulsar = SimplePTA.Npulsar
Nsamples = 50000

AGWBuse = 10**-14
Awuse = 0.1
Aruse = 10**np.random.uniform(-18,-13, size=Npulsar)
gammaruse = np.random.uniform(1,7, size=Npulsar)

CorrData = SimplePTA.generate_data(Nsamples=Nsamples, AGWB=AGWBuse, Aw=Awuse, Ar=Aruse, gammar=gammaruse)

ExactCorr = Likelihood(SimplePTA).ttcorrmat(Aw=Awuse, AGWB=AGWBuse, Ar=Aruse, gammar=gammaruse)

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
plt.savefig('CorrExample.jpg', bbox_inches='tight')