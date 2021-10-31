#!/usr/bin/env python
from ase.calculators.calculator import Calculator, all_changes
from minimulti.ioput.ifc_parser import IFCParser
from ase.io import read
from ase.units import Hartree
import numpy as np
import scipy.linalg as sl
import copy
import matplotlib.pyplot as plt
from minimulti.utils.supercell import SupercellMaker
from minimulti.utils.symbols import symbol_number
from itertools import product
from minimulti.scdm.rebase import rebase_wfn, rebase_H
from minimulti.unfolding.unfolder import Unfolder
from minimulti.unfolding.plotphon import plot_band_weight
from scipy.sparse import csr_matrix
from minimulti.abstract.datatype import Rijv
from ase.dft.kpoints import monkhorst_pack
from minimulti.scdm.scdmk import WannierScdmkBuilder, occupation_func, WannierProjectedBuilder
from minimulti.ioput.ifc_netcdf import read_ifc_from_netcdf


class Lattice(Calculator):
    implemented_properties = ['energy', 'forces']
    'Properties calculator can handle (energy, forces, ...)'

    nolabel = True
    default_parameters = {}
    'Default parameters'

    def __init__(self,
                 ifc=None,
                 restart=None,
                 ignore_bad_restart_file=False,
                 label=None,
                 atoms=None,
                 ref_atoms=None,
                 **kwargs):
        super(Lattice,
              self).__init__(restart=restart,
                             ignore_bad_restart_file=ignore_bad_restart_file,
                             label=label,
                             atoms=atoms,
                             **kwargs)
        #self.atoms=None
        if ifc is not None:
            self.total_ifc = ifc
        else:
            self.total_ifc = {}
        self.sr_ifc = None
        self.dd_ifc = None
        self.ref_atoms = copy.deepcopy(ref_atoms)
        self.ref_positions = ref_atoms.get_positions()

        self.natoms = len(self.ref_atoms)
        self.atoms = atoms
        self.energy = 0
        self.forces = np.zeros((self.natoms, 3), dtype=float)
        self._ifc_gamma = None
        self._has_ifc_gamma = False
        self.all_term_ifc = None

        self._extra_term = {}

        self.basis_positions = np.zeros([self.natoms * 3, 3])
        self.basis_positions[::3, :] = self.ref_atoms.get_scaled_positions()
        self.basis_positions[1::3, :] = self.ref_atoms.get_scaled_positions()
        self.basis_positions[2::3, :] = self.ref_atoms.get_scaled_positions()

    @classmethod
    def from_ifc_ncfile(cls, fname, **kwargs):
        atoms, Rlist, ifc_vallist = read_ifc_from_netcdf(fname)
        natom3=len(atoms)*3
        print(natom3)
        ifc = Rijv(shape=(natom3, natom3))
        for R, val in zip(Rlist, ifc_vallist):
            ifc[tuple(R)] = val
        return cls(ifc=ifc, ref_atoms=atoms, atoms=atoms, **kwargs)

    def calculation_required(self, atoms, properties):
        return True

    @property
    def natom(self):
        return self.natoms

    def initialize(self, atoms=None):
        if atoms is not None:
            self.atoms = atoms

    def get_variable_labels(self):
        sns = symbol_number(self.ref_atoms).keys()
        labels = [
            "%s_%s" % (v[0], v[1]) for v in product(sns, ('dx', 'dy', 'dz'))
        ]
        return labels

    def add_term(self, term, name):
        self._extra_term[name] = term

    def make_supercell(self, sc_matrix=None, supercell_maker=None):
        if supercell_maker is None:
            spm = SupercellMaker(sc_matrix=sc_matrix)
        else:
            spm = supercell_maker
        sc_ref_atoms = spm.sc_atoms(self.ref_atoms)
        sc_ifc = self.total_ifc.make_supercell(scmaker=spm)
        return Lattice(ifc=sc_ifc, ref_atoms=sc_ref_atoms)

    def get_cell(self):
        return self.ref_atoms.get_cell()

    def read_ifc_file(self, fname):
        parser = IFCParser(atoms=self.ref_atoms, fname=fname)
        self.total_ifc = parser.get_total_ifc()

    def get_ifc_gamma(self, sparse=False):
        """
        This is not IFC at Gamma, but a IFC with periodic boundary condition at gamma
        """
        self.ifc_gamma = np.zeros((self.natoms * 3, self.natoms * 3),
                                  dtype=float)
        for key, val in self.total_ifc.items():
            self.ifc_gamma += val
        self._has_ifc_gamma = True
        if sparse:
            self.ifc_gamma = csr_matrix(self.ifc_gamma)
        return self.ifc_gamma

    @property
    def dx(self):
        return self.get_dx()

    def get_dx(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        dx = atoms.get_positions() - self.ref_atoms.get_positions()
        return dx

    def calculate(
        self,
        atoms=None,
        properties=['energy', 'forces'],
        system_changes=all_changes,
    ):
        super(Lattice, self).calculate(atoms, properties, system_changes)
        self.results = {
            'energy': self.energy,
            'forces': self.forces,
            'stress': np.zeros(6),
            'dipole': np.zeros(3),
            'charges': np.zeros(len(atoms)),
            'magmom': 0.0,
            'magmoms': np.zeros(len(atoms))
        }
        if not self._has_ifc_gamma:
            self.get_ifc_gamma()

        #if atoms is not None:
        #    self.initialize(atoms=atoms)
        dx = self.get_dx(atoms=atoms)
        forces = -np.dot(dx.flatten(), self.ifc_gamma).reshape(
            (self.natoms, 3))
        forces -= np.sum(forces, axis=0) / self.natoms
        for name, term in self._extra_term.items():
            f = term.get_forces(displacement=dx.flatten())
            forces += f

        energy = -0.5 * np.sum(forces * dx)
        self.energy = energy
        self.forces = forces
        self.results['forces'] = forces
        self.results['energy'] = energy

    def get_allterm_ifc(self):
        ifc = Rijv(shape=(self.natoms * 3, self.natoms * 3))
        ifc += self.total_ifc
        for term in self._extra_term.values():
            d = term.get_ifc()
            ifc += d
        self.all_term_ifc = ifc

    def get_dynamical_matrix_q(self, qpoint, add_phase=True):
        if self.all_term_ifc is None:
            self.get_allterm_ifc()
        ifc = self.all_term_ifc
        natoms = len(self.ref_atoms)
        masses = np.repeat(self.ref_atoms.get_masses(), 3)
        #masses = np.ones(natoms*3)
        mat = np.zeros((natoms * 3, natoms * 3), dtype='complex')
        #ifc=copy.deepcopy(self.total_ifc)
        massij = np.sqrt(np.outer(masses, masses))
        for R, m in ifc.items():
            if add_phase:
                dis = self.basis_positions[
                    None, :, :] - self.basis_positions[:, None, :] + np.array(
                        R)[None, None, :]
                phase = np.exp(2j * np.pi *
                               np.einsum('ijk, k -> ij', dis, qpoint))
                mat += m * phase
            else:
                mat += np.exp(2j * np.pi * np.dot(R, qpoint)) * m
        mat = mat / massij
        #mat = (mat + mat.T.conj()) / 2
        return np.array(mat)

    def get_phonon_q(self, qpoint, add_phase=True, ham=False):
        dynmat = self.get_dynamical_matrix_q(qpoint, add_phase=add_phase)
        evals, evecs = sl.eigh(dynmat)
        if ham:
            return dynmat, evals, evecs
        else:
            return evals, evecs

    def solve_phonon(self,
                     qpoints,
                     add_phase=True,
                     ham=False,
                     transform_evals=True):
        nqpoints = len(qpoints)
        Hks = []
        evals = np.zeros((nqpoints, self.natoms * 3), dtype='float')
        evecs = np.zeros((nqpoints, self.natoms * 3, self.natoms * 3),
                         dtype='complex')
        for iqpt, qpt in enumerate(qpoints):
            if ham:
                Hk, es, vs = self.get_phonon_q(qpt,
                                               ham=True,
                                               add_phase=add_phase)
            else:
                es, vs = self.get_phonon_q(qpt, add_phase=add_phase)
            if ham:
                Hks.append(Hk)
            evals[iqpt, :] = es
            evecs[iqpt, :, :] = vs

        if transform_evals:
            s = np.sign(evals)
            v = np.sqrt(evals * s)
            evals = s * v * 15.633302 * 33
        if ham:
            return np.array(Hks), evals, evecs
        else:
            return evals, evecs

    def plot_phonon_band(self,
                         kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                            [0.5, 0.5, 0], [0, 0, 0],
                                            [.5, .5, .5]]),
                         knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                         supercell_matrix=None,
                         npoints=100,
                         color='red',
                         ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        evalues, evecs = self.solve_phonon(kpts)
        for i in range(3 * self.natoms):
            ax.plot(x, evalues[:, i], color=color, alpha=1)
        plt.axhline(0, linestyle='--', color='gray')
        ax.set_xlabel('q-point')
        ax.set_ylabel('Frequency (cm$^{-1}$)')
        #ax.set_ylabel('Energy (eV)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def unfold_phonon(self, kpts, sc_matrix):
        evals, evecs = self.solve_phonon(qpoints=kpts,
                                         add_phase=True,
                                         ham=False)
        positions = np.repeat(self.ref_atoms.get_scaled_positions(), 3, axis=0)
        # tbmodel: evecs[iband, ikpt, iorb]
        # unfolder: [ikpt, iorb, iband]
        sns = self.ref_atoms.get_chemical_symbols()
        labels = [
            "%s_%s" % (v[0], v[1]) for v in product(sns, ('x', 'y', 'z'))
        ]
        self.unf = Unfolder(
            cell=self.ref_atoms.cell,
            basis=labels,
            positions=positions,
            supercell_matrix=sc_matrix,
            #eigenvectors=np.swapaxes(evecs, 1, 2),
            eigenvectors=evecs,
            qpoints=kpts)
        return evals, self.unf.get_weights()

    def plot_unfolded_band(self,
                           kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                              [0.5, 0.5, 0], [0, 0, 0],
                                              [.5, .5, .5]]),
                           knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                           supercell_matrix=None,
                           npoints=300,
                           color='red',
                           ax=None):
        """
        plot the projection of the band to the basis
        """
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.ref_atoms.cell, npoints)
        kslist = np.array([x] * (self.natoms * 3))
        evals, wkslist = self.unfold_phonon(
            kpts, supercell_matrix)  #.T * 0.98 + 0.01
        wkslist = wkslist * 0.98 + 0.02
        ekslist = evals

        ax = plot_band_weight(kslist,
                              ekslist.T,
                              wkslist=wkslist.T,
                              efermi=None,
                              yrange=None,
                              output=None,
                              style='alpha',
                              color='blue',
                              axis=ax,
                              width=20,
                              xticks=None)
        for i in range(3 * self.natom):
            ax.plot(x, evals[:, i], color='gray', alpha=1, linewidth=0.1)
        ax.axhline(0.0, linestyle='--', color='gray')
        ax.set_ylabel('Frequency (THz)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def projwann(self,
                 mu,
                 sigma,
                 nwann,
                 ftype='unity',
                 kpts=[5, 5, 5],
                 anchors={(0.0, 0.0, 0.0): (1, 2, 3)},
                 Rgrid=None):
        kmesh = monkhorst_pack(kpts)
        evalues, evecs = self.solve_phonon(qpoints=kmesh,
                                           transform_evals=False)
        pwf = WannierProjectedBuilder(evals=evalues,
                                      wfn=evecs,
                                      has_phase=True,
                                      positions=self.basis_positions,
                                      kpts=kmesh,
                                      nwann=nwann,
                                      weight_func=occupation_func(ftype=ftype,
                                                                  mu=mu,
                                                                  sigma=sigma),
                                      Rgrid=Rgrid)
        pwf.set_projectors_with_anchors(anchors)
        lwf = pwf.get_wannier()
        return lwf

    def scdmk(self,
              mu,
              sigma,
              nwann,
              ftype='unity',
              kpts=[5, 5, 5],
              anchors={(0.0, 0.0, 0.0): (1, 2, 3)},
              selected_cols=None,
              Rgrid=None):
        kmesh = monkhorst_pack(kpts)
        evalues, evecs = self.solve_phonon(qpoints=kmesh,
                                           transform_evals=False)

        scdmk = WannierScdmkBuilder(evals=evalues,
                                    wfn=evecs,
                                    has_phase=True,
                                    positions=self.basis_positions,
                                    kpts=kmesh,
                                    nwann=nwann,
                                    weight_func=occupation_func(ftype=ftype,
                                                                mu=mu,
                                                                sigma=sigma),
                                    Rgrid=Rgrid)
        scdmk.set_selected_cols(selected_cols)
        scdmk.set_anchors(anchors)
        lwf = scdmk.get_wannier()
        return lwf

    def plot_wann_band(self,
                       lwf,
                       kvectors=np.array([[0, 0, 0], [0.5, 0,
                                                      0], [0.5, 0.5, 0],
                                          [0, 0, 0], [.5, .5, .5]]),
                       knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                       supercell_matrix=None,
                       npoints=100,
                       color='red',
                       ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        Hks, evalues, evecs = self.solve_phonon(kpts,
                                                ham=True,
                                                add_phase=False)
        for i in range(evalues.shape[1]):
            ax.plot(x, evalues[:, i], color='green', alpha=0.8)

        evals, _ = lwf.solve_all(kpts)
        s = np.sign(evals)
        v = np.sqrt(evals * s)
        evals = s * v * 15.633302 * 33
        for iband in range(evals.shape[1]):
            ax.plot(x, evals[:, iband], marker='.', color='blue', alpha=0.3)
        ax.axhline(0.0, linestyle='--', color='gray')
        ax.set_ylabel('Frequency ($cm^-1$)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')

        return ax
