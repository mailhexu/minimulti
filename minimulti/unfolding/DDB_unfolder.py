#!/usr/bin/env python
from .plotphon import plot_band_weight
from .phonon_unfolder import phonon_unfolder
import sys
import matplotlib.pyplot as plt
from ase.dft.kpoints import get_special_points, bandpath
from ase.build import bulk
from ase.data import atomic_masses
from ase.io import write
import numpy as np
has_abipy = False
try:
    import abipy.abilab as abilab
    has_abipy = True
except:
    has_abipy = False


def displacement_cart_to_evec(displ_cart,
                              masses,
                              scaled_positions,
                              qpoint=None,
                              add_phase=True):
    """
    displ_cart: cartisien displacement. (atom1_x, atom1_y, atom1_z, atom2_x, ...)
    masses: masses of atoms.
    scaled_postions: scaled postions of atoms.
    qpoint: if phase needs to be added, qpoint must be given.
    add_phase: whether to add phase to the eigenvectors.
    """
    if add_phase and qpoint is None:
        raise ValueError('qpoint must be given if adding phase is needed')
    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    evec = displ_cart * m
    if add_phase:
        phase = [
            np.exp(-2j * np.pi * np.dot(pos, qpoint))
            for pos in scaled_positions
        ]
        phase = np.kron(phase, [1, 1, 1])
        evec *= phase
        evec /= np.linalg.norm(evec)
    return evec


def DDB_unfolder(
    DDB_fname,
    kpath_bounds,
    sc_mat,
    knames=None,
    dipdip=1,
    ngqpt=None,
    ax=None,
):
    """

    :param DDB_fname: DDB file name
    :param kpath_bounds: list of the high symmetry k-points path.
    :param sc_mat: supercell matrix. Non-diagonal matrices are allowed.
    :param knames: Names of the high symmetry kpoints, e.g. ['$\Gamma$', 'X', 'R']
    :param dipdip: whether to use dipole dipole interaction.
    :param ax: The matplotlib ax. If ax=None, it will be generated and returned.

    """
    if not has_abipy:
        raise Exception("Please install abipy to use DDB unfolder")
    DDB = abilab.abiopen(DDB_fname)
    struct = DDB.structure
    atoms = DDB.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]

    sc_kpath_bounds = np.dot(kpath_bounds, sc_mat)

    phbst, phdos = DDB.anaget_phbst_and_phdos_files(
        nqsmall=0,
        asr=1,
        chneut=1,
        dipdip=dipdip,
        verbose=1,
        ngqpt=ngqpt,
        #ngqpt=[4, 4, 4],
        ndivsm=10,
        lo_to_splitting=True,
        qptbounds=sc_kpath_bounds,
        dipquad=0,
        quadquad=0,
    )
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    # positions=np.kron(scaled_positions,[1,1,1])

    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            phmode = phbst.get_phmode(qpt, ibranch)
            evals[iqpt, ibranch] = phmode.freq
            evec = displacement_cart_to_evec(phmode.displ_cart,
                                             masses,
                                             scaled_positions,
                                             qpoint=qpt,
                                             add_phase=True)
            evecs[iqpt, :, ibranch] = evec

    uf = phonon_unfolder(atoms, sc_mat, evecs, qpoints, phase=False)
    weights = uf.get_weights()
    x = np.arange(nqpts)
    freqs = evals
    xpts = []
    for ix, xx in enumerate(x):
        for q in sc_kpath_bounds:
            if np.sum((np.array(qpoints[ix]) - np.array(q)) **
                      2) < 0.00000001 and ix not in xpts:
                xpts.append(ix)
    if knames is None:
        knames = [str(k) for k in kpath_bounds]
    ax = plot_band_weight([list(x)] * freqs.shape[1],
                          freqs.T * 8065.6,
                          weights[:, :].T * 0.95 + 0.01,
                          xticks=[knames, xpts],
                          style='alpha')
    return ax


def nc_unfolder(fname,
                sc_mat,
                kx=None,
                knames=None,
                # ghost_atoms=None,
                plot_width=False,
                weight_multiplied_by=None):
    if not has_abipy:
        raise Exception("Please install abipy to use nc_unfolder")

    ncfile = abilab.abiopen(fname)
    struct = ncfile.structure
    atoms = ncfile.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]

    # print numbers
    # print cell
    # print scaled_positions

    # print kpath_bounds

    phbst = ncfile.phbands
    # phbst.plot_phbands()
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    # positions=np.kron(scaled_positions,[1,1,1])
    freqs = phbst.phfreqs
    displ_carts = phbst.phdispl_cart

    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            #phmode = ncfile.get_phmode(qpt, ibranch)
            # print(2)
            evals[iqpt, ibranch] = freqs[iqpt, ibranch]
            #evec=phmode.displ_cart *m
            #phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            #phase = np.kron(phase,[1,1,1])
            # evec*=phase
            #evec /= np.linalg.norm(evec)
            evec = displacement_cart_to_evec(displ_carts[iqpt, ibranch, :],
                                             masses,
                                             scaled_positions,
                                             qpoint=qpt,
                                             add_phase=True)
            evecs[iqpt, :, ibranch] = evec

    uf = phonon_unfolder(atoms,
                         sc_mat,
                         evecs,
                         qpoints,
                         phase=False,
                         # ghost_atoms=ghost_atoms
                         )
    write("atoms.vasp", atoms, vasp5=True, sort=True)
    weights = uf.get_weights()
    if plot_width:
        weights = (weights * (1.0 - weights))**(0.5)
    if weight_multiplied_by is not None:
        weights = weights * weight_multiplied_by
    x = np.arange(nqpts)
    freqs = evals
    ax = plot_band_weight([list(x)] * freqs.shape[1],
                          freqs.T * 8065.6,
                          weights[:, :].T * 0.98 + 0.000001,
                          xticks=[knames, kx],
                          style='alpha')
    return ax


def main():
    """example of how to use DDB_unfolder
    """
    # supercell matrix
    sc_mat = np.eye(3)*2
    # kpath in primitive cell
    kpoints = np.array([(0, 0, 0), (0, .5, 0), (0.5, 0.5, 0), [.5, .5, .5],
                        [0, 0, 0]])
    # Note the k.sc_mat should be used instead of k.
    DDB_unfolder(DDB_fname='out_DDB',
                 kpath_bounds=[np.dot(k, sc_mat) for k in kpoints],
                 sc_mat=sc_mat, knames=['$\Gamma$', 'X', 'M', 'R', '$\Gamma$'])
