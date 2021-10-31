import numpy as np
import matplotlib.pyplot as plt
from ase.dft.kpoints import bandpath
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
from minimulti.learn.polynomial import PolynomialDerivative
from minimulti.electron.plot import plot_band_weight
from minimulti.math import Pert
from minimulti.electron.ijR import *


class EPCData():
    def __init__(self,
                 ref_positions,
                 nphonon,
                 phonon_amplitude=None,
                 ijR_list=None,
                 ncoeff=0,
                 func=None):
        self.ref_positions = ref_positions
        self.nphonon = nphonon
        self.phonon_amplitude = phonon_amplitude
        self.ijR_list = ijR_list
        if ijR_list is not None:
            self.nbasis = ijR_list[0].nbasis
        self.coeffs = None
        self.ncoeff = ncoeff
        self.func = func

    def check(self, R=(0, 0, 0), i=None, j=None):
        # check consistency
        nbasis_list = np.array([x.nbasis for x in ijR_list])
        if not (nbasis_list == nbasis_list[0]).all():
            raise ValueError("not all models have same norb")
        nbasis = nbasis_list[0]

        if i is None:
            irange = range(nbasis)
        else:
            irange = [i]
        if j is None:
            jrange = range(nbasis)
        else:
            jrange = [j]
        for i in irange:
            for j in jrange:
                ydata = [m.data[tuple(R)][i, j] for m in ijR_list]
                plt.plot(self.phonon_amplitude, ydata)
        plt.show()

    def format_data(self, cutoff=1e-5):
        """
        format data so it can fit into sklearn fit.
        X: (nsample, nfeature) so each row is the feature vector.
        y: (nsamples, ntarget) so each row is the values for all ijR 
        nsamples is the number of configurations.
        """
        # R list
        RR = tuple(set(x.Rlist) for x in self.ijR_list)
        Rlist = set.union(*RR)
        Rlist = sorted(list(Rlist))
        ndata = len(self.ijR_list)
        nbasis = self.ijR_list[0].nbasis
        # for each R, build dok matrix
        # [[iR, i, j],[...]]
        self.Rlist = Rlist
        self.indlist = np.zeros((0, 3), dtype=int)
        # (nijR, ind) = (nsample, ntarget)
        self.vallist = np.zeros((ndata, 0))
        for iR, R in enumerate(self.Rlist):
            ms = np.array([ijR.data[R] for ijR in self.ijR_list])
            spp = np.zeros((nbasis, nbasis), dtype=bool)
            for m in ms:
                spp += (np.abs(m) > cutoff)
            irow, icol = np.where(spp)
            n = len(irow)
            self.indlist = np.vstack(
                [self.indlist,
                 np.array([[iR] * n, irow, icol]).T])
            # val: id: [ind_ijR, ind_matrix], ind_ms -> ntarget in sklearn fit.
            val = ms[:, irow, icol]
            self.vallist = np.hstack([self.vallist, val])
        return self.indlist, self.vallist

    def save(self, fname):
        with open(fname, 'wb') as myfile:
            d = {
                'ref_positions': self.ref_positions,
                'nbasis': self.nbasis,
                'nphonon': self.nphonon,
                'ncoeff': self.ncoeff,
                'Rlist': self.Rlist,
                'indlist': self.indlist,
                'model': self.model
            }
            pickle.dump(d, myfile)

    def gen_model(self, phonon_amplitude):
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        for R in self.coeffs[0]:
            m.data[R] = self.func(*[c[R] for c in self.coeffs])
        return m

    @staticmethod
    def load_fitting(fname):
        with open(fname, 'rb') as myfile:
            d = pickle.load(myfile)
        cls = EPCData(ref_positions=d['ref_positions'], nphonon=d['nphonon'])
        cls.nbasis = d['nbasis']
        cls.ncoeff = d['ncoeff']
        cls.Rlist = d['Rlist']
        cls.indlist = d['indlist']
        cls.model = d['model']
        return cls

    def predict_ijR(self, amplitude):
        y = self.model.predict([amplitude])[0]
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        istart = 0
        for k, g in groupby(self.indlist, lambda x: x[0]):
            g = np.array(list(g))
            R = tuple(self.Rlist[k])
            mat = np.zeros((self.nbasis, self.nbasis))
            iend = istart + len(g)
            val = y[istart:iend]
            mat = coo_matrix((val, (g[:, 1], g[:, 2])),
                             shape=(self.nbasis, self.nbasis),
                             dtype=float).todense()
            istart = iend
            m.data[R] = mat
        return m

    def predict_model(self, amplitude, ref_model, spin=True):
        m = self.predict_ijR(amplitude)
        if spin:
            m = m.to_spin_polarized()
        return ijR_to_model(copy.deepcopy(ref_model), m)

    def predict_H1(self, amplitude, i):
        """
        predict first order derivative of Hamiltonian. and generate a ijR object.
        """
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        dpoly = PolynomialDerivative(
            self.model.named_steps['polynomialfeatures'],
            self.model.named_steps['linearregression'].coef_, i)
        y = dpoly.predict([amplitude])

        istart = 0
        for k, g in groupby(self.indlist, lambda x: x[0]):
            g = np.array(list(g))
            R = tuple(self.Rlist[k])
            mat = np.zeros((self.nbasis, self.nbasis))
            iend = istart + len(g)
            val = y[istart:iend]
            mat = coo_matrix((val, (g[:, 1], g[:, 2])),
                             shape=(self.nbasis, self.nbasis),
                             dtype=float).todense()
            istart = iend
            m.data[R] = mat
        return m

    def polyfit(self, degree=6, fname='polymodel.pickle'):
        poly = PolynomialFeatures(degree=degree)
        model = make_pipeline(poly, LinearRegression())
        self.indlist, self.vallist = self.format_data()
        X_train, X_test, y_train, y_test = train_test_split(
            self.phonon_amplitude, self.vallist, test_size=0.2)
        model.fit(self.phonon_amplitude, self.vallist)
        score = model.score(self.phonon_amplitude, self.vallist)
        print(score)
        score = model.score(X_test, y_test)
        print(score)
        self.model = model
        return model

    def fit_data(self, phonon_amplitude, ijR_list):
        func = self.func
        ncoeff = self.ncoeff
        # check consistency
        nbasis_list = np.array([x.nbasis for x in ijR_list])
        if not (nbasis_list == nbasis_list[0]).all():
            raise ValueError("not all models have same norb")
        nbasis = nbasis_list[0]

        # build list of R
        RR = [x.Rlist for x in ijR_list]
        Rset = set(RR[0])
        for R in RR:
            Rset.intersection_update(R)
        Rlist = tuple(Rset)

        R = (0, 0, 0)
        for i in [1]:
            for j in range(nbasis):
                ydata = [m.data[R][i, j] for m in ijR_list]
                plt.plot(phonon_amplitude, ydata)
        plt.show()

        coeffs = [ijR(nbasis) for i in range(ncoeff)]
        xdata = np.array(phonon_amplitude).T
        for R in Rlist:
            for i in range(nbasis):
                for j in range(nbasis):
                    ydata = np.real([m.data[R][i, j] for m in ijR_list])
                    coeff, pcov = curve_fit(func,
                                            xdata,
                                            ydata,
                                            p0=[0.001] * ncoeff,
                                            method='trf')
                    perr = np.linalg.norm(np.diag(pcov))
                    if perr > 1:
                        print("coeff:", coeff)
                        print(pcov)
                        print(perr)
                        print(xdata)
                        print(ydata)
                        print(R, i, j)
                        plt.plot(phonon_amplitude, ydata)
                        plt.show()
                    for icoeff in range(ncoeff):
                        coeffs[icoeff].data[R][i, j] = coeff[icoeff]
        for icoeff in range(ncoeff):
            coeffs[icoeff].save('coeff_%s.nc' % icoeff)
        self.nbasis = nbasis
        self.coeffs = [dict(c.data) for c in coeffs]
        return coeffs

    def load_coeffs(self, coeffs=None, coeff_files=None, coeff_path=None):
        if coeff_path is not None:
            coeff_files = [
                os.path.join(coeff_path, 'coeff_%s.nc' % i)
                for i in range(self.ncoeff)
            ]
        if coeff_files is not None:
            self.coeffs = []
            for fname in coeff_files:
                c = ijR.load_ijR(fname)
                self.coeffs.append(dict(c.data))

    def gen_model(self, phonon_amplitudes):
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        for R in self.coeffs[0]:
            m.data[R] = self.func(*[c[R] for c in self.coeffs])
        return m



def epc_shift(kpts, evecs, dham, evals=None, onsite=True, order=1):
    onsite_energies = np.diag(dham[(0, 0, 0)])
    if not onsite:
        np.fill_diagonal(dham[(0, 0, 0)], 0.0)
    nband, nkpt, norb = evecs.shape
    ret = np.zeros((nband, nkpt), dtype=float)
    for ik, k in enumerate(kpts):
        Hk = np.zeros((norb, norb), dtype=complex)
        for R, HR in dham.items():
            phase = np.exp(2.0j * np.pi * np.dot(k, R))
            Hk += HR * phase
        Hk += Hk.T.conjugate()
        #for ib in range(nband):
        #    ret[ib, ik] = np.real(
        #        np.vdot(evecs[ib, ik, :], np.dot(Hk, evecs[ib, ik, :])))
        evecs_ik = evecs[:, ik, :].T
        if order == 1:
            #ret[:, ik] = np.diag(evecs_ik.T.conj().dot(Hk).dot(evecs_ik))
            ret[:, ik] = Pert.Epert1(evecs_ik, Hk)
        elif order == 2 and evals is not None:
            ret[:, ik] = Pert.Epert2(evals[:, ik], evecs_ik, Hk, norb)
        else:
            raise ValueError(
                "Error in calculating epe shift, for order should be 1 or 2, and for order2, evals should be given."
            )

    if not onsite:
        np.fill_diagonal(dham[(0, 0, 0)], onsite_energies)
    return ret


class EPC(object):
    def __init__(self, epc_dict=None, norb=None):
        if epc_dict is not None:
            self._epc = epc_dict
            for key, val in epc_dict.items():
                self._norb = val.shape[0]
                break

        elif norb is not None:
            self._epc = {}
            self._norb = norb

    def to_spin_polarized(self):
        norb = self._norb * 2
        epc = dict()
        for R, val in self._epc.items():
            epc[R] = np.zeros((norb, norb), dtype=float)
            epc[R][::2, ::2] = val
            epc[R][1::2, 1::2] = val
        return EPC(epc)

    @property
    def epc(self):
        return self._epc

    @property
    def norb(self):
        return self._norb

    def add_term(self, R, i, j, val):
        if R not in self._epc:
            self._epc[R] = np.zeros((self._norb, self._norb), dtype=float)
        self._epc[R][i, j] = val

    def add_term_R(self, R, mat):
        self._epc[R] = mat

    def get_band_shift(self, kpts, evecs, evals=None, order=1, onsite=True):
        return epc_shift(
            kpts, evecs, self._epc, evals=evals, order=order, onsite=onsite)

    def plot_epc_fatband(self,
                         kpts,
                         evals,
                         evecs,
                         k_x,
                         X,
                         order=1,
                         onsite=True,
                         xnames=None,
                         width=5,
                         show=False,
                         efermi=None,
                         axis=None,
                         **kwargs):

        wks = self.get_band_shift(
            kpts, evecs, evals=evals, order=order, onsite=onsite)
        kslist = [k_x] * self._norb
        ekslist = evals
        axis = plot_band_weight(
            kslist,
            ekslist,
            wkslist=wks,
            efermi=efermi,
            yrange=None,
            style='color',
            color='blue',
            width=width,
            axis=axis,
            **kwargs)
        for i in range(self._norb):
            axis.plot(k_x, evals[i, :], color='gray', linewidth=0.3)
        axis.set_ylabel('Energy (eV)')
        axis.set_xlim(k_x[0], k_x[-1])
        axis.set_xticks(X)
        if xnames is not None:
            axis.set_xticklabels(xnames)
        for x in X:
            axis.axvline(x, linewidth=0.6, color='gray')
        if show:
            plt.show()
        return axis
