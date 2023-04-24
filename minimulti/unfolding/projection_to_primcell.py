#!/usr/bin/env python3

import numpy as np
import abipy.abilab as abilab
from abipy.abilab import PhbstFile
from minimulti.unfolding.DDB_unfolder import atomic_masses, displacement_cart_to_evec
from ase.io import read, write


from ase.io import read, write
import numpy as np


def read_phonon(fname):
    """
    Read the phonon information from file.
    """
    ncfile = abilab.abiopen(fname)
    struct = ncfile.structure
    atoms = ncfile.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]
    phbst = ncfile.phbands
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    freqs = phbst.phfreqs
    displ_carts = phbst.phdispl_cart

    for iqpt, qpt in enumerate(qpoints):
        print(iqpt, qpt)
        for ibranch in range(nbranch):
            # phmode = ncfile.get_phmode(qpt, ibranch)
            # print(2)
            evals[iqpt, ibranch] = freqs[iqpt, ibranch]
            # evec=phmode.displ_cart *m
            # phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            # phase = np.kron(phase,[1,1,1])
            # evec*=phase
            # evec /= np.linalg.norm(evec)
            evec = displacement_cart_to_evec(
                displ_carts[iqpt, ibranch, :], masses, scaled_positions, qpoint=qpt, add_phase=True)
            evecs[iqpt, :, ibranch] = evec
    return atoms, evals, evecs, freqs


def match_supercell_to_primcell(patom, satom):
    """
    match the supercell atom to primitive cell atom.
    For each atom in the supercell, find the indices and R vector in the primcell.
    returns:
    ==================
    indices:
    Rs:
    """
    pxcart = patom.get_positions()
    ppos = patom.get_scaled_positions()
    ppos -= ppos[0]
    cell = patom.get_cell()

    sxcart = satom.get_positions()
    spos = cell.scaled_positions(sxcart-pxcart[0])

    iprims = []
    Rs = []

    for i, s in enumerate(spos):
        d = (s-ppos)
        res = d-np.round(d)
        norms = np.linalg.norm(res, axis=1)
        iprim = np.argmin(norms)
        R = np.round(d[iprim])
        if norms[iprim] > 0.1:
            print("Warning: the matching is probably bad.")
        iprims.append(iprim)
        Rs.append(R)
    return iprims, Rs


def build_map_matrix(iprim):
    return np.kron(np.array(iprim)*3, [1, 1, 1]) + np.array([0, 1, 2]*len(iprim))


def get_projection(patoms, satoms, pevecs, sevecs):
    iprim, R = match_supercell_to_primcell(patoms, satoms)


def test():
    patoms, pevals, pevecs, pfreqs = read_phonon("./pristine/o_PHBST.nc")
    #write("pristine.vasp", patoms, vasp5=True)

    #atoms, evals, evecs = read_phonon("./pristine2/o_PHBST.nc")
    #write("pristine2.vasp", atoms, vasp5=True)

    #atoms, evals, evecs = read_phonon("./1T_P1/run_PHBST.nc")
    #write("1T_P1.vasp", atoms, vasp5=True)

    satoms, sevals, sevecs, sfreqs = read_phonon("./2H_P1/o_PHBST.nc")
    write("2H_P1.vasp", satoms, vasp5=True)
    print(sevecs.shape)

    iprim, R = match_supercell_to_primcell(patoms, satoms)
    print(np.reshape(iprim, [6, 16]))
    ind_m = build_map_matrix(iprim)
    print(ind_m)

    # evecs indices: iqpt, ibasis, ibranch
    t_evecs = pevecs[:, ind_m, :]/4

    print(np.linalg.norm(sevecs[0, 0, :]))
    print(np.linalg.norm(t_evecs[0, :, 0]))

    proj = (t_evecs[0].T.conj() @ sevecs[0])**2
    proj = np.real(proj)
    print(proj.shape)
    print(np.real(proj))
    print(np.sum(proj, axis=0))
    sm = np.sum(proj, axis=0)

    result = np.zeros((289, 20))
    result[0, 1:19] = pfreqs*8065.6
    result[1:, 0] = sfreqs*8065.6
    result[1:, 1:-1] = proj.T
    result[1:, -1] = sm
    np.savetxt("projection_2HP1.txt", result, fmt="%6.2f")


test()
