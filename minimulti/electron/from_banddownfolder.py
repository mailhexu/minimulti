import os
import copy
import json
import numpy as np
from banddownfolder.scdm.lwf import LWF
from minimulti.electron.Hamiltonian import atoms_model
from minimulti.electron.basis2 import BasisSet, Basis
from minimulti.utils.symbol import symbol_number
from minimulti.electron.ijR import ijR
from minimulti.utils.supercell import SupercellMaker
from ase.atoms import Atoms
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import pickle
import numba


def read_basis(fname):
    """
    read the information of Wannier functions from BandDownfolder json file
    """
    bdict = dict()
    with open(fname) as myfile:
        c = json.load(myfile)
    atoms = Atoms(c['chemical_symbols'],
                  scaled_positions=c['atom_xred'],
                  cell=c['cell'])
    orbs = c['Orb_names']
    Efermi = c['Efermi']
    bset = BasisSet()
    sdict = symbol_number(atoms)
    for orb in orbs:
        sn, label, spin = orb.split('|')
        label = label.split('Z')[0][1:]
        site = sdict[sn]
        bset.append(Basis(site=site, label=label, spin=spin, index=0))
    bset.set_atoms(atoms)
    return bset, Efermi


def model_from_band_downfolder(path, lwf_fname, json_fname='Downfold.json'):
    """
    read wannier functions and set to minimulti electron model.
    lwf_fname: the name of the netcdf file containing Wannier functions
    json_fname: json filename for the downfolding
    """
    bset, Efermi = read_basis(os.path.join(path, json_fname))
    lwf = LWF.load_nc(os.path.join(path, lwf_fname))
    return lwf_to_model(bset, lwf, Efermi)


def lwf_to_model(bset, lwf, Efermi):
    """
    utility to set the 
    bset: basis set
    lwf: Wannier function object.
    Efermi: Fermi energy
    """
    atoms = bset.atoms
    bset = bset.add_spin_index()
    model = atoms_model(atoms=atoms, basis_set=bset, nspin=2)
    hop = {}
    for iR, R in enumerate(lwf.Rlist):
        R = tuple(R)
        hr = np.zeros((
            lwf.nwann * 2,
            lwf.nwann * 2,
        ), dtype='complex')
        hr[::2, ::2] = lwf.hoppings[tuple(R)] / 2
        hr[1::2, 1::2] = lwf.hoppings[tuple(R)] / 2
        hop[R] = hr

    en = np.zeros(lwf.nwann * 2, dtype=float)
    en[::2] = lwf.site_energies + Efermi
    en[1::2] = lwf.site_energies + Efermi
    model._hoppings = hop
    model._site_energies = en
    return model


def run_model(model: atoms_model, U, J, plot=False):
    model.set(nel=4)
    model.set_kmesh([4, 4, 4])
    model.set_Hubbard_U(Utype='Liechtenstein',
                        Hubbard_dict={'V': {
                            'U': U,
                            'J': J
                        }})
    model.scf_solve()
    model.save_result(pfname=f'Results_Liech/result_U{U:.2f}_J{J:.2f}.pickle')
    if plot:
        model.plot_band(kvectors=[[0, 0, 0], [0, 0.5, 0], [.5, .5, 0],
                                  [.5, 0, 0], [0, 0, 0], [0, 0.25, 0.5],
                                  [0.5, 0.25, 0.5], [0.5, 0, 0], [0, 0, 0],
                                  [0., 0.25, -0.5], [.5, 0.25, -0.5]],
                        knames='GYCZGBDZGAE',
                        shift_fermi=True)
        figname = f'Results_Liech/result_U{U:.2f}_J{J:.2f}.png'
        plt.savefig(figname)
        plt.show()
        plt.close()


def make_supercell(Rlist,
                   HwannR,
                   xred,
                   sc_matrix=None,
                   smaker=None,
                   func=None,
                   order=1):
    if smaker is None:
        smaker = SupercellMaker(sc_matrix)
    sxred = smaker.sc_pos(xred)
    from collections import defaultdict
    nbasis = HwannR.shape[1]
    ret = defaultdict(lambda: np.zeros(
        (nbasis * smaker.ncell, nbasis * smaker.ncell)))
    for c, cur_sc_vec in enumerate(
            smaker.sc_vec):  # go over all super-cell vectors
        for iR, R in enumerate(Rlist):
            sc_part, pair_ind = smaker._sc_R_to_pair_ind(tuple(R + cur_sc_vec))
            sc_part = tuple(sc_part)
            ii = c * nbasis
            jj = pair_ind * nbasis
            if func is None:
                f = 1
            else:
                f = func(R)**order
            ret[sc_part][ii:ii + nbasis, jj:jj + nbasis] += HwannR[iR] * f
    return sxred, ret


def set_Hwann_to_model(model, mat, Rlist):
    Rdict = {}
    for iR, R in enumerate(Rlist):
        Rdict[tuple(R)] = iR
    nwann = mat.shape[1]
    en = np.zeros(nwann * 2, dtype=float)
    site_energies = np.diag(mat[Rdict[(0, 0, 0)]])
    en[::2] = site_energies
    en[1::2] = site_energies
    hop = {}
    for iR, R in enumerate(Rlist):
        R = tuple(R)
        hr = np.zeros((
            nwann * 2,
            nwann * 2,
        ), dtype='complex')
        hr[::2, ::2] = mat[iR, :, :] / 2.0
        hr[1::2, 1::2] = mat[iR, :, :] / 2
        if tuple(R) == (0, 0, 0):
            np.fill_diagonal(hr, np.zeros(nwann * 2))
        hop[R] = hr
    model._hoppings = hop
    model._site_energies = en
