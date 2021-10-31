import sys
from math import log, exp
import numpy as np
from numba import vectorize, float64, jit

MAX_EXP_ARGUMENT = log(sys.float_info.max)


@vectorize([float64(float64, float64, float64)])
def fermi(e, mu, width):
    """
    the fermi function.
     .. math::
        f=\\frac{1}{\exp((e-\mu)/width)+1}

    :param e,mu,width: e,\mu,width
    """

    x = (e - mu) / width
    if x < MAX_EXP_ARGUMENT:
        return 1 / (1.0 + exp(x))
    else:
        return 0.0


#fermi = np.vectorize(fermi_s)


def density_matrix(eigenvals, eigenvecs, efermi, width, spin_split=True):
    """
    calculate the density matrix. (not in k space)
     .. math::
        \\rho(p,q)=\\sum_\\alpha A_{p,\\alpha} f(\\alpha-\\mu) A^*_{q,\\alpha}
    and the energy (if calc_E).
     .. math::
        E=\\frac{1}{2} (\\sum_\\alpha\\epsilon_\\alpha f(\\epsilon_\\alpha-\\mu) + \\sum_{qq} H_{pq}^0 \\rho_{qp})


    :param eigenvals: (np.ndarray ) energy levels. indexes (band)
    :param eigenvecs: (np.ndarray ) energy eigenvectors. indexes:(orb,band)
    :param efermi: fermi energy
    :param width: smearing width
    :param spin_split (bool): if true ,return (rho_up,rho_dn)
    :param calc_E: (bool) whether to calculate the Hartree Fock energy.
    :returns:
     rho(p,q)  indexes are [state,state], Note here, spin index are included in p.
    """
    f = fermi(eigenvals, efermi, width)
    rho = (eigenvecs * f).dot(eigenvecs.conj().transpose())
    if not spin_split:
        return rho
    else:
        return rho[::2, ::2], rho[1::2, 1::2]


@jit(nopython=True, fastmath=True, cache=True)
def rhok(occ_k, evec_k, kweight, rho):
    nstate, nband = evec_k.shape
    for i in range(nband):
        for j in range(nstate):
            tmp_ij = kweight * occ_k[i] * evec_k[j, i].conjugate()
            for k in range(nstate):
                rho[j, k] += (tmp_ij * (evec_k[k, i])).real


@jit(nopython=True, fastmath=True, cache=True)
def rhok_diag(occ_k, evec_k, kweight, rho):
    nstate, nband= evec_k.shape
    for i in range(nband):
        tmp = kweight * occ_k[i]
        for j in range(nstate):
            rho[j] += tmp * (evec_k[j, i] * evec_k[j, i].conjugate()).real


def density_matrix_kspace(
    eigenvecs,
    occupations,
    kweight,
    diag_only=True,
    rho_k=None,
    rho=None,
):
    """
    calculate the density matrix with multiple k.
     .. math::
        \\rho= \sum_k \\rho_k weight(k)

    :param eigenvecs: the eigenvec matrix. indexes are [band,kpt,orb,spin] or [band,kpt,orb]
    :param occupations: the occupation matrix. indexes are [band,kpt] the same as the eigenvals. Not the same as the eigenvecs.
    :param kweight: (ndarray) used if eigenvalues
    :param split_spin (bool)
    """
    nk, nspin, norb, nband = eigenvecs.shape
    nstate = norb * nspin
    rho[:] = 0.0
    for k in range(nk):
        occ_k = occupations[k, ispin]
        for ispin in range(nspin):
            evec_ks = eigenvecs[k, ispin]
            if rho_k is not None:
                if diag_only:
                    rhok(occ_ks, evec_ks, kweight[k], rho_k[k, ispin])
                    rhok_diag(occ_ks, evec_ks, kweight[k], rho[ispin])
                else:
                    rhok(occ_ks, evec_ks, kweight[k], rho_k[k, ispin])
                    rho[ispin] += rho_k[k, ispin]
            else:
                if diag_only:
                    rhok_diag(occ_ks, evec_ks, kweight[k], rho[ispin])
                else:
                    rhok(occ_ks, evec_ks, kweight[k], rho[ispin])
    return rho


def density_matrix_kspace_merge_spin(
    eigenvecs,
    occupations,
    kweight,
    diag_only=True,
    rho_k=None,
    rho=None,
):
    """
    calculate the density matrix with multiple k.
     .. math::
        \\rho= \sum_k \\rho_k weight(k)

    :param eigenvecs: the eigenvec matrix. indexes are [band,kpt,orb,spin] or [band,kpt,orb]
    :param occupations: the occupation matrix. indexes are [band,kpt] the same as the eigenvals. Not the same as the eigenvecs.
    :param kweight: (ndarray) used if eigenvalues
    """
    nk, nspin, norb, nband = eigenvecs.shape
    nso = norb * nspin
    if diag_only:
        rhos = np.zeros((nspin, norb), dtype=float)
    else:
        rhos = np.zeros((nspin, norb, norb), dtype=float)
    for k in range(nk):
        for ispin in range(nspin):
            occ_ks = occupations[k, ispin]
            evec_ks = eigenvecs[k, ispin]
            if rho_k is not None:
                if diag_only:
                    rhok(occ_ks, evec_ks, kweight[k], rho_k[k, ispin])
                    rhok_diag(occ_ks, evec_ks, kweight[k], rhos[ispin])
                else:
                    rhok(occ_ks, evec_ks, kweight[k], rho_k[k, ispin])
                    rhos[ispin] += rho_k[k, ispin]
            else:
                if diag_only:
                    rhok_diag(occ_ks, evec_ks, kweight[k], rhos[ispin])
                else:
                    rhok(occ_ks, evec_ks, kweight[k], rhos[ispin])
    if diag_only:
        if nspin == 2:
            rho = np.zeros((nso, ), dtype=float)
            rho[::2] = rhos[0]
            rho[1::2] = rhos[1]
        else:
            rho = rhos
    else:
        if nspin == 2:
            rho = np.zeros((nso, nso), dtype=float)
            rho[::2, ::2] = rhos[0]
            rho[1::2, 1::2] = rhos[1]
        else:
            rho = rhos
    return rho


class DensityMatrix(object):
    diag_only = False

    def __init__(self, norb):
        self.norb = norb
        self._rho = np.zeros((norb, norb), dtype=float)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        assert np.all(self.rho.shape == rho.shape)
        self._rho = rho

    def add_diag(self, drho):
        ids = np.diag_indices(self.norb)
        self._rho[ids] += drho

    @property
    def density(self):
        return np.diag(self.rho)

    @property
    def density_matrix(self):
        return self.rho

    def save(self, fname=None):
        np.save(fname, self.rho, allow_pickle=False)

    def load(self, fname=None):
        rho = np.load(fname, allow_pickle=False)
        if not (rho.shape == self.rho.shape):
            raise ValueError(
                f"Cannot read from pickle file {fname}. shape not consistent.")
        self.rho = rho

    def density_matrix_kspace(
        self,
        eigenvecs,
        occupations,
        kweight,
        rho_k=None,
    ):
        self.rho=density_matrix_kspace_merge_spin(eigenvecs,
                                         occupations,
                                         kweight,
                                         diag_only=False,
                                         rho_k=None,
                                         rho=self.rho)

    def spin_for_each_orbital(self):
        return self.density[::2] - self.density[1::2]

    def block(self, ids):
        return self.rho[np.ix_(ids, ids)]

    def block_diag(self, ind):
        return self.density[ind]

    def diag_i(self, i):
        return self.rho[i, i]


class DiagDensityMatrix(DensityMatrix):
    diag_only = True

    def __init__(self, norb):
        self.norb = norb
        self._rho = np.zeros((norb), dtype=float)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        assert self._rho.shape == rho.shape
        self._rho = rho

    @property
    def density(self):
        return self.rho

    @property
    def density_matrix(self):
        return np.diag(self.rho)

    def add(self, drho):
        self.rho += drho

    def add_diag(self, drho):
        self.rho += drho

    def density_matrix_kspace(
        self,
        eigenvecs,
        occupations,
        kweight,
        rho_k=None,
    ):
        self.rho=density_matrix_kspace_merge_spin(eigenvecs,
                                         occupations,
                                         kweight,
                                         diag_only=True,
                                         rho_k=None,
                                         rho=self._rho)

    def density_splitspin(self):
        return self.rho[::2], self.rho[1::2]

    def block(self, ids):
        return np.diag(self.rho[ids])

    def block_diag(self, ind):
        return self.rho[ind]

    def diag_i(self, i):
        return self.rho[i]


def gen_density_matrix(norb, diag_only):
    if diag_only:
        dm = DiagDensityMatrix(norb)
    else:
        dm = DensityMatrix(norb)
    return dm
