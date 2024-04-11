"""This module implements a minimal simulator for PTA residuals.

There is no attempt to actually model physically accurate TOAs (in any frame),
rather, given the specification of arbitrary TOAs and pulsar locations (or
a direct correlation matrix), residuals corresponding to a specified signal
can be generated.

These can then be quickly analyzed using the Likelihood class for the given PTA
in order to provide both approximate and exact LLs (for realisitic PTAs
the latter will be very slow) and the optimial statistic of [arXiv:1410.8256]."""

import numpy as np
import scipy.special as sps

class Pulsar:
    '''A container storing basic pulsar properties. Contains no actual data. '''
    def __init__(self, TOAs, location=None, name=None):
        self.TOAs = TOAs
        self.loc = location
        self.name = name

    @property
    def TOAs(self):
        return self._TOAs

    # Ensures that G projection matrix is recomputed if TOAs for the pulsar are changed
    @TOAs.setter
    def TOAs(self, value):
        self._TOAs = value
        self.Nobs = len(value)

        # Computes the 'design matrix' for a quadratic timing model given the TOAs.
        # TOAs are norminally imagined to be given in MJD format, but this should be irrelevant
        Nobs, sortedTOAs = len(self.TOAs), np.sort(self.TOAs)
        sortedTOAs = sortedTOAs - sortedTOAs[0] # convert from times to time differences

        Mmat = np.ones((Nobs, 3))
        Mmat[:,1] = sortedTOAs
        Mmat[:,2] = sortedTOAs**2
        
        Umat, sdiag, _ = np.linalg.svd(Mmat)

        self.G = Umat[:,3:]



class PTA:
    '''A PTA dataset generator. Takes a list of Pulsar objects as an input.
       Various frequency sampling parameters are set to default values at
       initialization, but can be manually changed at later points if so required.
         N.B. All frequencies are in units of 1/year.
              All TOA information is assumed to be in MJD format (units of days),
              while the residuals themselves are reported in seconds. '''
    def __init__(self, Pulsars, f_L=None, Delta_f=None, Nfreq=100, covmat=None):
        self.Npulsar = len(Pulsars)
        self.Pulsars = Pulsars

        # Internal constants for conversion
        self._dtoyr = 1/365.24
        self._dtos  = 24.*60.*60.
        self._yrtos = 365.24*self._dtos

        # Compute default generation frequency settings if not specified
        if f_L == None:
            # Default lowest frequency is 1/total observation time
            DeltaTobs = self._dtoyr * (max([psr.TOAs[-1] for psr in self.Pulsars]) - min([psr.TOAs[0] for psr in self.Pulsars]))
            self.f_L = 1./DeltaTobs
            self.Delta_f = self.f_L
        else:
            self.f_L = f_L
            self.Delta_f = Delta_f
        self.Nfreq = Nfreq

        # Compute all kernels for signal generation
        self.freqlist = self.Delta_f*np.arange(self.Nfreq) + self.f_L
        for psr in self.Pulsars:
            ftmat = np.outer(self.freqlist, self._dtoyr * psr.TOAs)
            psr.ftkernelf = self.Delta_f * np.cos(2*np.pi*ftmat)
            psr.ftkernelg = self.Delta_f * np.sin(2*np.pi*ftmat)


        # Compute angles and HD correlation matrix if not specified
        #  N.B. Requires Pulsar objects to have locations set
        if covmat is None:
            if all([psr.loc is not None for psr in self.Pulsars]):
                cosmat = np.zeros((self.Npulsar, self.Npulsar), dtype=np.float64)
                for i, psri in enumerate(self.Pulsars):
                    for j, psrj in enumerate(self.Pulsars[i:]):
                        if j == 0:
                            cosmat[i,i] = 1
                        else:
                            cosmat[i,i+j] = self._cosfromecliptic(psri.loc,psrj.loc)
                        
                # copy upper triangular part over diagonal
                cosmat += np.triu(cosmat,k=1).T
                self.cosmat = cosmat

                #HD covariance matrix for pulsars
                chimat = (1 - cosmat)/2
                self.covmat = 1.5 * sps.xlogy(chimat, chimat) - 0.25 * chimat + 0.5 + 0.5*np.eye(self.Npulsar)
            else:
                raise ValueError("Some pulsar locations missing, please provide manual Hellings-Downs covariance matrix!")
        else:
            self.covmat = covmat

    # Helper method for computation of covariance matrix from pulsar locations
    def _cosfromecliptic(self, loc1, loc2):
        ''' Computes cosine between two points specified by (longitude, latitude). '''
        dlong = (loc2[0] - loc1[0])
        dlat = (loc2[1] - loc1[1])
        haversine = np.sin(dlat/2)**2 + np.cos(loc1[1])*np.cos(loc2[1])*np.sin(dlong/2)**2
        
        return 1-2*haversine

    # to match libstempo, Aw should have units of microseconds, but time series have units of seconds      
    def generate_whitenoise(self, Nsamples, Aw=1):
        ''' Generates residual white noise for Nsamples, with amplitude Aw given in microseconds. '''
        if hasattr(Aw, '__len__'):
            testdatat = [1e-6 * Aw[i] * np.random.normal( size=(Nsamples, psr.Nobs ) ) for i, psr in enumerate(self.Pulsars)]
        else:
            testdatat = [1e-6 * Aw * np.random.normal( size=(Nsamples, psr.Nobs ) ) for psr in self.Pulsars]
        return testdatat

    # AGWB should be dimensionless, corresponds to the power in the strain at 1/yr
    def generate_SGWB(self,Nsamples,AGWB=1,gamma=13/3):
        ''' Generates a correlated GW signal with a given spectral index. '''
        # This is the list of cos and sin frequency amplitudes, sampled from a multivariate gaussian
        # Derived from Eqs. (34-35) of [arXiv:1410.8256]
        amplitudes = 1/np.sqrt(12*np.pi**2)*np.sqrt(1/self.Delta_f)*(self.freqlist)**(-(gamma)/2)

        # amplitudes have units of time^2
        normals = np.random.multivariate_normal(mean=np.zeros(self.Npulsar), cov=self.covmat, size = (Nsamples,self.Nfreq) )
        fklist = normals * amplitudes.reshape((1,-1,1))

        normals = np.random.multivariate_normal(mean=np.zeros(self.Npulsar), cov=self.covmat, size = (Nsamples,self.Nfreq) )
        gklist = normals * amplitudes.reshape((1,-1,1))

        # All frequency information in units of years, we want residual series in seconds
        testdatat = [self._yrtos * AGWB * (fklist[:,:,i] @ psr.ftkernelf + gklist[:,:,i] @ psr.ftkernelg) for i, psr in enumerate(self.Pulsars)]
        
        return testdatat 

    def generate_rednoise(self,Nsamples,Ar,gammar):  # Ar (gammar) must be a numpy array of length Nsample*Npulsar (Npulsar)
        ''' Generates red noise for all pulsars given a list of amplitudes and spectral indices, '''
        
        testdatat = [0] * self.Npulsar

        for i, psr in enumerate(self.Pulsars):
            # This is the list of cos and sin frequency amplitudes, sampled from a simple gaussian
            # Derived from Eqs. (34-35) of [arXiv:1410.8256]
            amplitudes = 1/np.sqrt(12*np.pi**2)*np.sqrt(1/self.Delta_f)*(self.freqlist)**(-(gammar[i])/2)

            normals = np.random.normal( size=(Nsamples, self.Nfreq ) )
            fklist = normals * amplitudes.reshape((1,-1))

            normals = np.random.normal( size=(Nsamples, self.Nfreq ) )
            gklist = normals * amplitudes.reshape((1,-1))

            # All frequency information in units of years, we want residual series in seconds
            testdatat[i] = self._yrtos*Ar[i]*(fklist @  psr.ftkernelf + gklist @  psr.ftkernelg)

        return testdatat 
    
    def generate_data(self,Nsamples,AGWB=1,Aw=1,Ar=0,gammar=1,gammaGW=13./3.):
        ''' Generates a complete set of residuals from various sources.'''
        testdatat_WN = self.generate_whitenoise(Nsamples,Aw)
        if np.min(Ar) > 0:
            testdatat_RN = self.generate_rednoise(Nsamples,Ar,gammar)
        else:
            testdatat_RN = [0] * self.Npulsar
        testdatat_GW = self.generate_SGWB(Nsamples,AGWB,gammaGW)

        testdatat=[ (testdatat_WN[i]+testdatat_RN[i]+testdatat_GW[i]) @ psr.G for i, psr in enumerate(self.Pulsars)]

        return testdatat



class Likelihood:
    '''Computes various likelihood approximations and derived functions.
       Takes in a PTA object for all pulsar timing information.'''
    def __init__(self, PTA):
        # The PTA object here is mainly a way to efficiently pass all the pulsar information we need
        # Projection matrices are accessible through Pulsars
        self.Npulsar = PTA.Npulsar
        self.Pulsars = PTA.Pulsars
        self.covmat = PTA.covmat

        # Internal constants for conversion
        self._dtoyr = 1/365.24
        self._dtos  = 24.*60.*60.
        self._yrtos = 365.24*self._dtos

        # Precompute TOA difference matrix to all pulsars for quick access
        # (Convert to units of years since all correlations are defined with units of 1/yr)
        self.tmat = [[self._dtoyr * np.abs(psri.TOAs.reshape(-1,1) - psrj.TOAs.reshape(1,-1))
                      for psrj in self.Pulsars] for psri in self.Pulsars]

        # Precompute canonical gamma=13/3 correlation matrix for expedited OS calcualation
        self.corrmatGW = [[self.covmat[i, j] * (psri.G.T @ self.analytic_corr(self.tmat[i][j]) @ psrj.G)
                           for j, psrj in enumerate(self.Pulsars)] for i, psri in enumerate(self.Pulsars)]

        # Internal conversion factors
        self.daytoyear = 1./365.24
        self.daytosecond = 24.*60.*60.
        self.yeartosecond=365.*self.daytosecond

    def tmat_func(self, k, kp):
      p1toas = self.TOA[k]
      p2toas = self.TOA[kp]

      return np.array([[np.abs(p1toas[i] - p2toas[j]) for i in range(len(p1toas))] for j in range(len(p2toas))])

    def analytic_corr(self, tau, gamma=13/3):
      ''' Return the analyic correlation between for a given time interval,
          (ignoring the regulator term, which (usually) doesn't matter after projection.) '''
      return self._yrtos**2/(12*np.pi**2) * (2*np.pi*tau)**(gamma-1) * sps.gamma(1-gamma) * np.sin(gamma*np.pi/2)

    def crosscorrmat(self, A=1, gamma=13/3, covmat=None):
        ''' Compute an array of correlation matrices on the TOAs for the spactial correlation specified in the PTA. '''
        # corrmat_arr = np.block([[A**2 * self.covmat[i, j] * (psri.G.T @ self.analytic_corr(self.tmat[i][j], gamma=gamma) @ psrj.G)
        #                          for j, psrj in enumerate(self.Pulsars)] for i, psri in enumerate(self.Pulsars)])
        if covmat == None:
            corrmat_arr = [[A**2 * self.covmat[i, j] * (psri.G.T @ self.analytic_corr(self.tmat[i][j], gamma=gamma) @ psrj.G)
                            for j, psrj in enumerate(self.Pulsars)] for i, psri in enumerate(self.Pulsars)]
        else:
            corrmat_arr = [[A**2 * covmat[i, j] * (psri.G.T @ self.analytic_corr(self.tmat[i][j], gamma=gamma) @ psrj.G)
                            for j, psrj in enumerate(self.Pulsars)] for i, psri in enumerate(self.Pulsars)]

        return corrmat_arr

    def ttcorrmat(self, Aw=0.1, AGWB=0, Ar=0, gammar=0, gamma=None):
        ''' Projected residual correlation matrix
            Aw is in units of [msec], AGWB/Ar is the amplitude at f = 1/yr '''

        # Start with all of the pulsar-to-pulsar covariance matrices
        if gamma == None:
            fullttcorrmat = [[AGWB**2 * self.corrmatGW[i][j] for j in range(self.Npulsar)] for i in range(self.Npulsar)]
        else:
            fullttcorrmat = self.crosscorrmat(AGWB, gamma)

        for i, psr in enumerate(self.Pulsars):
            # Add red noise contribution
            fullttcorrmat[i][i] += Ar[i]**2 * (psr.G.T @ self.analytic_corr(self.tmat[i][i], gammar[i]) @ psr.G)
            # Add white noise contribution
            if hasattr(Aw, '__len__'):
                fullttcorrmat[i][i] += (Aw[i]*1e-6)**2 * np.identity(psr.Nobs-3)
            else:
                fullttcorrmat[i][i] += (Aw*1e-6)**2 * np.identity(psr.Nobs-3)

        return fullttcorrmat
    
    def exactLL(self,t,Aw=0.1,AGWB=0,Ar=0,gammar=0,gamma=13/3): # t are the residuals
        fullttcorrmat = np.block(self.ttcorrmat(Aw=Aw,AGWB=AGWB,Ar=Ar,gammar=gammar,gamma=gamma))
        Ctotinvtemp = np.linalg.inv(fullttcorrmat)
        sign, logCdettemp = np.linalg.slogdet(fullttcorrmat)

        return 0.5*np.dot(t, Ctotinvtemp @ t) + 0.5*sign*logCdettemp

    def approxLL(self,t,Aw=0.1,AGWB=0,Ar=0,gammar=0,gamma=13/3):
        '''This gives the LL ignoring all pulsar-pulsar correlations. Need it for the noise-marginalized OS'''
        approxttcorrmat=np.block(self.ttcorrmat(Aw=Aw,AGWB=AGWB,Ar=Ar,gammar=gammar,gamma=gamma,covmat=np.identity(self.Npulsar)))
        Ctotinvtemp=np.linalg.inv(approxttcorrmat)
        sign,logCdettemp=np.linalg.slogdet(approxttcorrmat)

        return 0.5*np.dot(t,np.matmul(Ctotinvtemp,t)) + 0.5*sign*logCdettemp
    
    def optimalstatistic(self,t,Aw=0.1,AGWB=0,Ar=0,gammar=0,gamma=None):
        '''define the OS with some random choice for AGWB in the autocorrelators
           t should be in the format [[Nsamples, t_P1], [Nsamples, t_P2], ..., [Nsamples, t_PN]]'''
           #Follow definitions 40-49 in https://arxiv.org/pdf/1410.8256.pdf
        
        if gamma == None:
            Smat = self.corrmatGW
        else:
            Smat = self.crosscorrmat(A=1, gamma=gamma)

        # Compute the P matrices directly. Only need the diagonal part, so less overhead than asking for the full correlation matrix
        if hasattr(Aw, '__len__'):
            Pmat = [Ar[i]**2 * (psr.G.T @ self.analytic_corr(self.tmat[i][i], gammar[i]) @ psr.G) + AGWB**2 * Smat[i][i] + (Aw[i]*1e-6)**2 * np.identity(psr.Nobs-3)
                   for i, psr in enumerate(self.Pulsars)]
        else:
            Pmat = [Ar[i]**2 * (psr.G.T @ self.analytic_corr(self.tmat[i][i], gammar[i]) @ psr.G) + AGWB**2 * Smat[i][i] + (Aw*1e-6)**2 * np.identity(psr.Nobs-3)
                   for i, psr in enumerate(self.Pulsars)]
        invPmat = [np.linalg.inv(PI) for PI in Pmat]

        numer, denom = 0, 0
        Nsamples = len(t[0])
        for i in range(self.Npulsar):
            for j in range(i):
                kernel = invPmat[i] @ Smat[i][j] @ invPmat[j]
                numer += t[i].reshape(Nsamples,1,-1) @ kernel @ t[j].reshape(Nsamples,-1,1) # reshape to broadcast for multi-sample calculation
                denom += np.trace(kernel @ Smat[j][i])

        numer = np.squeeze(numer) # flatten broadcasting 

        # If the OS uncertainty is needed, it is 1/np.sqrt(denom)

        return numer/denom
