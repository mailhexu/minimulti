"""
Hamiltonian terms
"""
import numpy as np
from collections import defaultdict
from minimulti.constants import mu_0, Boltzmann
import scipy.sparse as ssp
import numba


class HamTerm(object):
    """
    Base class for Hamiltonian terms.
    """

    def __init__(self, ms=None):
        self.E = 0.0
        self.ms = ms
        if ms is not None:
            self.nmatoms = len(ms)
        self._hessian = None
        self._hessian_ijR = None

    def eff_field(self, S, Heff):
        r"""
        Hi = -1/ms_i * (\partial H/\partial Si)
        """
        raise NotImplementedError

    def calc_hessian(self):
        raise NotImplementedError()

    def hessian(self):
        r"""
        \partial^2 H/ (\paritial Si \partial Sj)
        """
        if self._hessian is None:
            self.calc_hessian()
        return self._hessian

    def calc_hessian_ijR(self):
        raise NotImplementedError()

    def hessian_ijR(self):
        if self._hessian_ijR is None:
            self.calc_hessian_ijR()
        return self._hessian_ijR

    def energy(self, S):
        raise NotImplementedError()


class SingleBodyTerm(HamTerm):
    def __init__(self, ms=None):
        super(SingleBodyTerm, self).__init__(ms=ms)

    def eff_field(self, S, Heff):
        raise NotImplementedError()

    def calc_hessian(self):
        self._hessian = 0

    def calc_hessian_ijR(self):
        self._hessian_ijR = {}

    def is_twobody_term(self):
        return False


class TwoBodyTerm(HamTerm):
    def __init__(self, ms=None):
        super(TwoBodyTerm, self).__init__(ms=ms)

    def eff_field(self, S, Heff):
        r"""
        \partial H/\partial Si
        """
        raise NotImplementedError()

    def is_twobody_term(self):
        return True


class ZeemanTerm(SingleBodyTerm):
    """
    $H_{zeeman}=-\\sum_i g_i \\mu_B \\vec{H}_i \\cdot \\vec{S}_i $
    """

    def __init__(self, H, ms=None):
        super(ZeemanTerm, self).__init__(ms)
        self.H = H

    def eff_field(self, S, Heff):
        r"""
        Hi = -1/ms_i * (\partial H/\partial Si)
        It is here because it is simpler than the form of jacobian. Therefore faster.
        """
        Heff[:, :] += self.H * self.ms[:, None]


class UniaxialMCATerm(TwoBodyTerm):
    """
    """

    def __init__(self, k1, k1dir, ms):
        super(UniaxialMCATerm, self).__init__(ms=ms)
        self.Ku = np.array(k1)  # Ku list of nspins.
        # normalize direction.
        direction = np.array(k1dir)
        self.e = direction / (np.linalg.norm(direction, axis=1)[:, None])
        assert (self.e.shape == (self.nmatoms, 3))
        assert (self.Ku.shape == ms.shape)

    def func_i(self, S, i):
        return -self.Ku[i] * np.dot(S[i], self.e[i])**2

    def eff_field(self, S, Heff):
        Heff[:, :] -= self.hessian().dot(S.reshape(3 * self.nmatoms)).reshape(
            self.nmatoms, 3)

    def calc_hessian(self):
        self._hessian = ssp.lil_matrix(
            (self.nmatoms * 3, self.nmatoms * 3), dtype=float)
        for i in range(self.nmatoms):
            self._hessian[i * 3:i * 3 + 3, i * 3:i * 3 + 3] = (
                -2.0 * self.Ku[i]) * np.outer(self.e[i], self.e[i])
        self._hessian = ssp.csr_matrix(self._hessian)
        return self._hessian

    def calc_hessian_ijR(self):
        self._hessian_ijR = {}
        for i in range(self.nmatoms):
            self._hessian_ijR[(i, i, (
                0, 0,
                0))] = (-2.0 * self.Ku[i]) * np.outer(self.e[i], self.e[i])
        return self._hessian_ijR


class HomoUniaxialMCATerm(SingleBodyTerm):
    """
    Homogenous Uniaxial Magnetocrystaline Anistropy
    """

    def __init__(self, Ku, direction, ms=None):
        super(HomoUniaxialMCATerm, self).__init__(ms=ms)
        self.Ku = Ku
        self.e = np.array(direction) / np.linalg.norm(direction)

    def func_i(self, S, i):
        return -self.Ku * np.dot(S[i], self.e)**2

    def jacobian_i(self, S, i):
        return -2.0 * self.Ku * np.dot(S[i], self.e) * self.e

    def eff_field(self, S, Heff):
        Heff += 2.0 * self.Ku * np.outer(
            np.einsum('ij,j,i->i', S, self.e, 1), self.e)


class ExchangeTerm(TwoBodyTerm):
    """
    exchane interaction in Heissenberg model
    """

    def __init__(self, Jdict, ms=None, sparse_matrix_form=True, pbc=[1, 1, 1]):
        """
        J is given as a dict of {(i, j, R): val},
         where R is a tuple, val is a scalar.
        """
        super(ExchangeTerm, self).__init__(ms=ms)
        self.Jdict = Jdict
        Jmat = defaultdict(float)
        for key, val in self.Jdict.items():
            i, j, R = key
            if np.all(R * (1 - np.array(pbc)) == 0):
                Jmat[(i, j)] += val
        self.ilist, self.jlist = np.array(tuple(Jmat.keys()), dtype='int').T
        self.vallist = np.array(tuple(Jmat.values()))
        self.jac = np.zeros((self.nmatoms, 3))
        self.nij = self.vallist.shape[0]
        self.Heff = np.zeros((self.nmatoms, 3))

    def eff_field(self, S, Heff):
        S.shape = (self.nmatoms * 3)
        Heff.shape = (self.nmatoms * 3)
        Heff -= 2.0 * self.hessian().dot(S)
        S.shape = (self.nmatoms, 3)
        Heff.shape = (self.nmatoms, 3)
        return self.jac

    def calc_hessian(self):
        self._hessian = ssp.lil_matrix(
            (self.nmatoms * 3, self.nmatoms * 3), dtype=float)
        for i, j, val in zip(self.ilist, self.jlist, self.vallist):
            self._hessian[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= np.eye(3) * val

        self._hessian = ssp.csr_matrix(self._hessian)
        return self._hessian

    def calc_hessian_ijR(self):
        self._hessian_ijR = {}
        for key, val in self.Jdict.items():
            i, j, R = key
            self._hessian_ijR[(i, j, R)] = -np.eye(3) * val
        return self._hessian_ijR


class DMITerm(TwoBodyTerm):
    """
    Dzyaloshinskii-Moriya interaction.
    $H_{DM} = -\\sum {i<j} \\vec{D}_{ij} \\cdot \\S_i        imes \\S_j$
    """

    def __init__(self, ddict, ms):
        """
        J is given as a dict of {(i, j, R): val},
         where R is a tuple, val is a scalar.
        """
        super(DMITerm, self).__init__(ms=ms)
        self.ddict = ddict
        Dmat = defaultdict(float)
        for key, val in self.ddict.items():
            i, j, R = key
            Dmat[(i, j)] += np.array(val)

        self.ilist, self.jlist = np.array(tuple(Dmat.keys()), dtype='int').T
        self.vallist = np.array(tuple(Dmat.values()))
        self.jac = np.zeros((self.nmatoms, 3))
        self.nij = self.vallist.shape[0]
        self.Heff = np.zeros((self.nmatoms, 3))

    def eff_field(self, S, Heff):
        S.shape = self.nmatoms * 3
        jac = 2.0 * self.hessian().dot(S)
        S.shape = (self.nmatoms, 3)
        jac.shape = (self.nmatoms, 3)
        Heff[:, :] -= jac
        return jac

    def calc_hessian(self):
        self._hessian = ssp.lil_matrix(
            (self.nmatoms * 3, self.nmatoms * 3), dtype=float)
        for i, j, val in zip(self.ilist, self.jlist, self.vallist):
            self._hessian[i * 3:i * 3 + 3, j * 3:j * 3 + 3] += np.array(
                [[0, val[2], -val[1]], [-val[2], 0, val[0]],
                 [val[1], -val[0], 0]])
        self._hessian = ssp.csr_matrix(self._hessian)
        return self._hessian

    def calc_hessian_ijR(self):
        self._hessian_ijR = {}
        for key, val in self.ddict.items():
            i, j, R = key
            self._hessian_ijR[(i, j, R)] = np.array([[0, val[2], -val[1]],
                                                     [-val[2], 0, val[0]],
                                                     [val[1], -val[0], 0]])
        return self._hessian_ijR


class DipDip(TwoBodyTerm):
    """
    Dipolar interaction.
    TODO Note that Model.positions is reduced coordinates.
    """

    def __init__(self):
        pass
