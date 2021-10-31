"""PDOS"""

import numpy as np
import matplotlib.pyplot as plt
import numba


@numba.njit()
def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * \
            np.exp(-x**2 / 2.0 / sigma**2)


@numba.njit()
def _calc_dos(evals, evecs, sigma, nband, kpts, nspin, kweights, elist, pdos):
    evecs2 = np.abs(evecs)**2
    for ib in range(nband):
        for ik, i in enumerate(kpts):
            for ispin in range(nspin):
                w = kweights[ik]
                for ie, e in enumerate(elist):
                    a = gaussian(evals[ik, ispin, ib] - e, sigma)
                    pdos[ispin, :, ie] += evecs2[ik, ispin, :, ib] * a * w


class PDOS(object):
    def __init__(self, kpts, kweights, sigma, evals, evecs, emin, emax, nedos,
                 nspin, efermi):
        self.kpts = kpts
        self.kweights = kweights
        self.sigma = sigma
        self.evals = evals
        self.evecs = evecs
        self.nkpt, self.nspin, self.norb, self.nband = self.evecs.shape
        self.emin = emin
        self.emax = emax
        self.nedos = nedos
        #evecs <band, kpt, orb>
        #evals <band, kpt>
        self.elist = np.linspace(emin, emax, nedos)
        self.nspin = nspin
        self.pdos = np.zeros((self.nspin, self.norb, self.nedos),
                             dtype='float')
        self.efermi=efermi
        self._calc_dos()

    def _calc_dos(self):
        _calc_dos(self.evals, self.evecs, self.sigma, self.nband, self.kpts,
                  self.nspin, self.kweights, self.elist, self.pdos)

    def get_energy(self):
        return self.elist

    def get_pdos(self, isite=None, ispin=None):
        if isite is None:
            return self.pdos
        else:
            return self.pdos[ispin, isite, :]

    def get_total_dos(self):
        return np.sum(self.pdos, axis=(0, 1))

    def save_txt(self, fname):
        a=np.vstack([self.elist-self.efermi] + [self.pdos[i] for i in range(self.nspin)]).T
        np.savetxt(fname, a, header=f"energy, spinup, spindown\n  efermi={self.efermi}, nspin={self.nspin}, nedos={self.nedos} ")

    def save_pickle(self, fname):
        with open(fname, 'wb') as myfile:
            myfile.write({'nspin':self.nspin, 
                          'nedos': self.nedos,
                          'energy': self.elist, 
                          'pdos': self.pdos, 
                          'efermi':self.efermi})
                
                

class WDOS(object):
    def __init__(self, kpts, kweights, sigma, evals, weights, emin, emax,
                 nedos):
        self.kpts = kpts
        self.kweights = kweights
        self.sigma = sigma
        self.evals = evals
        self.nband, self.nkpt = self.evals.shape
        self.emin = emin
        self.emax = emax
        self.nedos = nedos
        #evals <band, kpt>
        self.elist = np.linspace(emin, emax, self.nedos)
        self.dx = self.elist[1] - self.elist[0]
        self.wdos = np.zeros(self.nedos, dtype='float')
        self.idos = np.zeros(self.nedos, dtype='float')
        self.weights = weights
        self._calc_dos()

    def _calc_dos(self):
        for ib in range(self.nband):
            for ik, i in enumerate(self.kpts):
                w = self.kweights[ik]
                for ie, e in enumerate(self.elist):
                    a = gaussian(self.evals[ib, ik] - e, self.sigma)
                    self.wdos[ie] += self.weights[ib, ik] * a * w

        for ie, e in enumerate(self.elist):
            self.idos[ie] = self.idos[ie - 1] + self.wdos[ie] * self.dx

    def get_energy(self):
        return self.elist

    def get_wdos(self):
        return self.wdos

    def get_idos(self):
        return self.idos
