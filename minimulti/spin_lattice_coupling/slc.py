"""
This module define the coupling between spin and lattice.
"""
import numpy as np
import numba
from scipy.sparse import dok_matrix, csr_matrix, bsr_matrix, coo_matrix
from minimulti.utils.supercell import SupercellMaker
from minimulti.abstract.datatype import IFClike, COOlike, Rijv
from minimulti.spin_lattice_coupling.fit_forces import read_Oiju_netcdf

from netCDF4 import Dataset
import pickle as pickle
from ase.units import eV, J
eVperJ = eV / J


@numba.njit(fastmath=True, cache=True)
def Oiju_force(vallist, ndata, ind, S, Sref, force):
    """
    ind: matrix of shape (nnz, 3)
    O: array [nnz, natom*3 ]
    """
    force[:] = 0.0
    for ii in range(ndata):
        i, j, u = ind[ii]
        Sijm1 = np.dot(S[i, :], S[j, :]) - np.dot(Sref[i, :], Sref[j, :])
        force[u] = vallist[ii] * Sijm1


#@numba.njit(fastmath=True, cache=True)
def Oiju_bfield(vallist, ndata, ind, displacement, S, bfield):
    for ii in range(ndata):
        i, j, u = ind[ii]
        bfield[i, :] += 2.0 * eVperJ * vallist[ii] * displacement[u] * S[j, :]


@numba.njit(fastmath=True, cache=True)
def Oiju_deltaJ(vallist, ndata, ind, displacement, deltaJ):
    for ii in range(ndata):
        i, j, u = ind[ii]
        deltaJ[i, j] += vallist[ii] * displacement[u]


@numba.njit(fastmath=True, cache=True)
def Tijuv_derivatives(vallist, nnz, ind, S, displacement, forces, delta_J,
                      delta_ifc):
    """
    O: array [nnz, natom*3, natom*3]
    """
    forces[:] = 0.0
    delta_J[:, :] = 0.0
    delta_ifc[:, :] = 0.0
    tmp = 0.0
    for ii in range(nnz):
        i, j, u, v = ind[ii]
        val = vallist[ii]
        tmp = val * (np.dot(S[i, :], S[j, :]) - 1.0)
        delta_ifc[u, v] -= tmp
        forces[u] += tmp * displacement[v]
        delta_J[i, j] += 0.5 * eVperJ * val * displacement[u] * displacement[v]
    return forces


class SLCTerm(object):
    @property
    def displacement(self):
        return self._displacement.reshape((self.natom, 3))

    @displacement.setter
    def displacement(self, val):
        if len(val.shape) == 1:
            self._displacement = val
        elif len(val.shape) == 2:
            self._displacement = val.reshape(self.natom * 3)

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, val):
        if len(val.shape) == 1:
            self._S = val.reshape(self.nspin, 3)
        elif len(val.shape) == 2:
            self._S = val


class OijuTerm(SLCTerm):
    def __init__(
            self,
            nspin,
            natom,
            ndata,
            ilist,
            jlist,
            ulist,
            Rjlist,
            Rulist,
            vallist,
            Sref=None,
    ):
        self.nspin = nspin
        self.natom = natom
        self.ndata = ndata
        self.ilist = ilist
        self.jlist = jlist
        self.ulist = ulist
        self.ind = np.array([ilist, jlist, ulist], dtype=int).T
        self.Rjlist = Rjlist
        self.Rulist = Rulist
        self.vallist = vallist

        self._forces = np.zeros((self.natom * 3), dtype=float)
        self._bfield = np.zeros((self.nspin, 3), dtype=float)
        if Sref is None:
            self.Sref = np.zeros((self.nspin, 3), dtype=float)
            self.Sref[:, 2] = 1.0
        else:
            self.Sref = Sref

    def make_supercell(self, sc_matrix=None, supercell_maker=None):
        if sc_matrix is None and supercell_maker is None:
            raise ValueError(
                "One of sc_matrix or supercell_maker should be inputed.")
        elif sc_matrix is not None:
            supercell_maker = SupercellMaker(sc_matrix)
        sc_nspin = self.nspin * supercell_maker.ncell
        sc_natom = self.natom * supercell_maker.ncell
        sc_ilist = supercell_maker.sc_index(self.ilist, self.nspin)
        sc_jlist, sc_Rjlist = supercell_maker.sc_jR(self.jlist, self.Rjlist,
                                                    self.nspin)
        sc_ulist, sc_Rulist = supercell_maker.sc_jR(self.ulist, self.Rulist,
                                                    self.natom * 3)
        sc_vallist = supercell_maker.sc_trans_invariant(self.vallist)
        sc_Oiju = OijuTerm(
            nspin=sc_nspin,
            natom=sc_natom,
            ndata=len(sc_ilist),
            ilist=sc_ilist,
            jlist=sc_jlist,
            ulist=sc_ulist,
            Rjlist=sc_Rjlist,
            Rulist=sc_Rulist,
            vallist=sc_vallist)
        return sc_Oiju

    def get_forces(self, S, displacement):
        self._forces.shape = self.natom * 3
        self._forces[:] = 0.0
        self.S=S
        Oiju_force(self.vallist, self.ndata, self.ind, self.S, self.Sref,
                   self._forces)
        self._forces.shape = (self.natom, 3)
        self._forces -= np.average(self._forces, axis=0)
        return self._forces

    def eff_field(self, S, Heff, displacement=None):
        if displacement is None:
            displacement = self.displacement
        else:
            self.displacement = displacement
        Oiju_bfield(self.vallist, self.ndata, self.ind, displacement.flatten(),
                    S, Heff)

    def get_ifc(self, S=None, displacement=None):
        return dict()

    @staticmethod
    def read_netcdf(fname):
        (nspin, natom, nnz, ilist, jlist, ulist, Rjlist, Rulist,
         vallist) = read_Oiju_netcdf(fname)
        return OijuTerm(
            nspin=nspin,
            natom=natom,
            ndata=nnz,
            ilist=ilist,
            jlist=jlist,
            ulist=ulist,
            Rjlist=Rjlist,
            Rulist=Rulist,
            vallist=vallist)

    def plot_Oiju_distribution(self):
        pass

@numba.njit(fastmath=True)
def exch_Heff(ilist, jlist, vallist, S, Heff):
    for ind in range(len(ilist)):
        i = ilist[ind]
        j = jlist[ind]
        val = vallist[ind]
        Heff[i, :] -= val * S[j, :] * 2.0


class TijuvTerm(SLCTerm):
    def __init__(self, natom, nspin, parameter):
        self.natom = natom
        self.nspin = nspin
        self.T_ijR = parameter
        self._nz_ij = set()
        self._nz_uv = set()
        self._ijR_to_COO()

        self._displacement = np.zeros(natom * 3)
        self._S = np.zeros((self.nspin, 3))

        self._forces = np.zeros((self.natom, 3), dtype='float')
        self._eff_field = np.zeros((self.nspin, 3), dtype='float')

    def read_netcdf(self, fname):
        pass

    def to_data(self):
        ilist = []
        jlist = []
        ulist = []
        vlist = []
        vallist = []
        for key, val in self.T_ijR.items():
            i, j, Rj = key
            for ii in range(3):
                for Rv, uv in val.items():
                    uvmat = coo_matrix(uv)

                    for u, v, d in zip(uvmat.row, uvmat.col, uvmat.data):
                        if abs(d) > 1e-8:
                            ilist.append(i * 3 + ii)
                            jlist.append([j * 3 + ii, Rj[0], Rj[1], Rj[2]])
                            ulist.append([u, 0, 0, 0])
                            vlist.append([v, Rv[0], Rv[1], Rv[2]])
                            vallist.append(d)

        ilist = np.array(ilist)
        jlist = np.array(jlist)
        ulist = np.array(ulist)
        vlist = np.array(vlist)
        vallist = np.array(vallist)
        return ilist, jlist, ulist, vlist, vallist

    def write_netcdf(self, fname):
        """
        write netcdf
        """
        ilist, jlist, ulist, vlist, vallist = self.to_data()
        snnz = len(vallist)
        natom3 = self.natom * 3
        ds = Dataset(fname, 'w')
        ds.createDimension("spin_lattice_Tijuv_number_of_entries", snnz)
        ds.createDimension("natom3", natom3)
        ds.createDimension("nspin", self.nspin)
        ds.createDimension("nspin3", self.nspin * 3)
        ds.createDimension("three", 3)
        ds.createDimension("four", 4)

        snnz = "spin_lattice_Tijuv_number_of_entries"

        ilist += 1
        jlist[:, 0] += 1
        ulist[:, 0] += 1
        vlist[:, 0] += 1

        v_ilist = ds.createVariable(
            varname="spin_lattice_Tijuv_ilist",
            datatype='i4',
            dimensions=(snnz, ))
        v_jlist = ds.createVariable(
            varname="spin_lattice_Tijuv_jlist",
            datatype='i4',
            dimensions=(snnz, "four"))
        v_ulist = ds.createVariable(
            varname="spin_lattice_Tijuv_ulist",
            datatype='i4',
            dimensions=(snnz, "four"))
        v_vlist = ds.createVariable(
            varname="spin_lattice_Tijuv_vlist",
            datatype='i4',
            dimensions=(snnz, "four"))

        v_vallist = ds.createVariable(
            varname='spin_lattice_Tijuv_valuelist',
            datatype='f8',
            dimensions=(snnz))

        direction_spins = ds.createVariable(
            varname="direction_spins",
            datatype='f8',
            dimensions=("nspin", "three"))

        direction_spins.setncatts({"long_name": u"direction of all spins"})

        # long names and units
        v_ilist.setncatts({
            "long_name":
            u"index i in Tijuv with xyz index included, starts from 1"
        })

        v_jlist.setncatts({
            "long_name":
            u"index j and Rj in Tijuv, j starts from 1, Rj has three numbers"
        })

        v_ulist.setncatts({
            "long_name":
            u"index u and Ru in Tijuv, u starts from 1, Ru has three numbers"
        })

        v_vlist.setncatts({
            "long_name":
            u"index v and Rv in Tijuv, v starts from 1, Rv has three numbers"
        })

        v_vallist.setncatts({
            "long_name":
            u"spin lattice coupling term Tijuv value list",
            "unit":
            "eV/Angstrom^2"
        })

        ds['spin_lattice_Tijuv_ilist'][:] = ilist
        ds['spin_lattice_Tijuv_jlist'][:] = jlist
        ds['spin_lattice_Tijuv_ulist'][:] = ulist
        ds['spin_lattice_Tijuv_vlist'][:] = vlist
        ds['spin_lattice_Tijuv_valuelist'][:] = vallist
        ds['direction_spins'][:] = np.array([[0.0,0.0,1.0]])
        ds.close()

    @staticmethod
    def load_pickle(fname, natom, nspin):
        with open(fname, 'rb') as myfile:
            J2dict = pickle.load(myfile)
        return TijuvTerm(natom, nspin, J2dict)

    def _ijR_to_COO(self):
        """
        T: a COO matrix of dense matrix
        """
        T_dok = {}
        for key_ijR, IFCdict in self.T_ijR.items():
            i, j, Rj = key_ijR
            self._nz_ij.add((i, j))

            if (i, j) not in T_dok:
                data_ij = np.zeros(
                    (self.natom * 3, self.natom * 3), dtype='float')
            else:
                data_ij = T_dok[(i, j)]
            for Rv, val in IFCdict.items():
                data_ij += val
            T_dok[(i, j)] = data_ij

        data = []
        for i, j in self._nz_ij:
            data.append(csr_matrix(T_dok[(i, j)]).todense())
        # COO matrix with CSR matrix as elements.
        self.T = COOlike(
            shape=(self.nspin, self.nspin),
            indices=tuple(self._nz_ij),
            data=data)

    @property
    def parameter(self):
        return self.T

    def make_supercell(self, sc_matrix=None, supercell_maker=None):
        if supercell_maker is None:
            smaker = SupercellMaker(sc_matrix)
            n_sc = int(round(np.linalg.det(sc_matrix)))
        else:
            smaker = supercell_maker
            n_sc = smaker.ncell
        sc_nspin = self.nspin * n_sc
        sc_natom = self.natom * n_sc

        T_sc = IFClike(
            genfunc=lambda: Rijv(shape=(sc_natom * 3, sc_natom * 3), sparse=False, dtype=float))

        natom3 = self.natom * 3
        for iRsc, Rsc in enumerate(smaker.R_sc):
            for key_ijR, val_ijR in self.T_ijR.items():
                i, j, Rj = key_ijR
                ip = smaker.sc_i_to_sci(i, iRsc, self.nspin)
                jp, Rjp = smaker.sc_jR_to_scjR(
                    j=j, R=tuple(Rj), Rv=tuple(Rsc), n_basis=self.nspin)

                for R, mat in val_ijR.items():
                    ind_R = np.array(R, dtype=int)
                    #sc_part, pair_ind = smaker._sc_R_to_pair_ind(
                    #    tuple(ind_R + Rsc))
                    #mat[np.abs(mat)<1e-3]=0
                    coomat = coo_matrix(mat)
                    for u, v, val in zip(coomat.row, coomat.col, coomat.data):
                        up = smaker.sc_i_to_sci(u, iRsc, natom3)
                        vp, Rvp = smaker.sc_jR_to_scjR(
                            j=v, R=tuple(R), Rv=tuple(Rsc), n_basis=natom3)
                        T_sc[(ip, jp, Rjp)][tuple(Rvp)][up, vp] += val
        return TijuvTerm(natom=sc_natom, nspin=sc_nspin, parameter=T_sc)

    def get_delta_J(self, displacement=None):
        """
        displacment: (natom*3 array)
        Note that both in dense form and sparse form, the expression is the same!
        """
        if displacement is None:
            displacement = self._displacement
        # if ifc is in dense matrix form.
        #dJ = 0.5*np.dot(np.dot(self.T.data, displacement), displacement)
        displacement = np.reshape(displacement, self.natom * 3)
        dJ = np.zeros((self.T.nnz), dtype=float)
        for ij, d in enumerate(self.T.data):
            dJ[ij] = 0.5 * eVperJ * np.dot(d.dot(displacement), displacement)
        return dJ

    def eff_field(self, S, Heff, displacement=None):
        if displacement is None:
            displacement = self._displacement
        else:
            self.displacement = displacement
        exch_Heff(self.T.row, self.T.col, self.get_delta_J(displacement), S,
                  Heff)

    def get_ifc(self, S=None, displacement=None):
        """
        Note that we have Rj=0, Ru=0, only Rv!=0.
        Thus, Phi(u, v, Rv)-> -Tijuv SiSj
        """
        if S is None:
            S = self._S
        ret = Rijv(
            shape=(self.natom * 3, self.natom * 3), sparse=False, dtype=float)
        for key_ijR, IFCdict in self.T_ijR.items():
            i, j, R = key_ijR
            Sijm1 = np.dot(S[i, :], S[j, :]) - 1.0
            ret -= IFCdict * Sijm1
        return ret

    def get_forces(self, displacement, S=None):
        if S is None:
            S = self._S
        self._forces.shape = self.natom * 3
        self._forces[:] = 0.0
        for i, j, data in zip(self.T.row, self.T.col, self.T.data):
            Sijm1 = np.dot(S[i, :], S[j, :]) - 1.0
            self._forces[:] += Sijm1 * data.dot(displacement)
        self._forces.shape = (self.natom, 3)
        avg_F = np.average(self._forces, axis=0)
        self._forces -= avg_F
        return self._forces


class SpinLatticeCoupling(object):
    """SpinLatticeCoupling"""

    def __init__(
            self,
            spin_model=None,
            lattice_model=None,
            natom=None,
            ms=None,
            Sref=None,
    ):
        """__init__"""

        if spin_model is not None:
            self.spin_model = spin_model
            self.ms = self.spin_model._ham.ms
            self.nspin = len(self.ms)
        elif ms is not None:
            self.ms = ms
            self.nspin = len(ms)
        else:
            raise ValueError(
                "at least one of spin_model or ms should be given.")

        if lattice_model is not None:
            self.lattice_model = lattice_model
            #self.lattice_model.add_term(self, name='Tijuv')
            self.natom = self.lattice_model.natom
        elif natom is not None:
            self.natom = natom
        else:
            raise ValueError(
                "at leat one of lattice_model and natom should be given")
        self.Sref = Sref

        self.terms = {}

        self._forces = np.zeros((self.natom, 3), dtype='float')
        self._eff_Hfield = np.zeros((self.nspin, 3), dtype='float')

        self._S = np.zeros((self.nspin, 3), dtype='float')
        self._displacement = np.zeros((self.natom * 3), dtype='float')

    def add_term(self, term, name):
        self.terms[name] = term
        self.terms[name].Sref = self.Sref

    #@profile
    def make_supercell(self,
                       sc_matrix=None,
                       supercell_maker=None,
                       sc_spin_model=None,
                       sc_lattice_model=None):
        if supercell_maker is not None:
            smaker = supercell_maker
        else:
            smaker = SupercellMaker(sc_matrix)
        sc_natom = self.natom * smaker.ncell
        sc_ms = smaker.sc_trans_invariant(self.ms)

        sc_Sref = np.zeros((len(sc_ms), 3), dtype=float)
        sc_Sref[:, 2] = np.real(smaker.phase([0.5, 0.5, 0.5]))

        slc = SpinLatticeCoupling(
            spin_model=sc_spin_model,
            lattice_model=sc_lattice_model,
            ms=sc_ms,
            natom=sc_natom,
            Sref=sc_Sref)
        for key, val in self.terms.items():
            slc.add_term(val.make_supercell(supercell_maker=smaker), name=key)
        print("supercell made")
        return slc

    @property
    def displacement(self):
        return self._displacement.reshape((self.natom, 3))

    @displacement.setter
    def displacement(self, val):
        if len(val.shape) == 1:
            self._displacement = val
        elif len(val.shape) == 2:
            self._displacement = val.reshape(self.natom * 3)
        for key, val in self.terms.items():
            val.displacement = self._displacement

    @property
    def S(self):
        return self._S.reshape((self.nspin, 3))

    @S.setter
    def S(self, val):
        if len(val.shape) == 1:
            self._S = val
        elif len(val.shape) == 2:
            self._S = val.reshape(self.nspin * 3)
        for key, val in self.terms.items():
            val.S = self._S

    def get_delta_J(self, displacement=None):
        self.delta_J = 0.0
        for term in self.terms.values():
            self.delta_J += term.get_delta_J(displacement=displacement)
        return self.delta_J

    def eff_field(self, S, Heff, displacement=None):
        if displacement is None:
            displacement = self.displacement
        else:
            self.displacement = displacement
        for term in self.terms.values():
            term.eff_field(S=S, displacement=displacement, Heff=Heff)

    def get_forces(self, displacement, S=None):
        if S is None:
            S = self.S
        else:
            self.S = S
        self._forces[:, :] = 0.0
        for name, term in self.terms.items():
            self._forces = term.get_forces(displacement=displacement, S=S)
        return self._forces

    def get_ifc(self):
        print("Here")
        ifc = Rijv(shape=(self.natom * 3, self.natom * 3))
        for term in self.terms.values():
            ifc += term.get_ifc()
        return ifc
