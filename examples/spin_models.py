#!/usr/bin/env python

from ase.atoms import Atoms
from minimulti.spin.hamiltonian import SpinHamiltonian
from minimulti.spin.mover import SpinMover
from minimulti.spin.qsolver import QSolver
from minimulti.spin.builder import *

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from minimulti.constants import mu_B, eV
import matplotlib.colors as colors
import os
meV = eV * 1e-3

from minimulti.spin.hist_file import read_hist
import ipyvolume as ipv

def plot_3d_vector(positions, vectors, length=0.1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax=fig.gca()
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        length=length,
        normalize=False,
        pivot='middle',
        cmap='seismic')
    plt.show()


def plot_2d_vector(positions, vectors, show_z=True, length=0.1, ylimit=None, marker_size=50):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    #ax.streamplot(x, y, u, v,  linewidth=1, cmap=plt.cm.inferno,
    #    density=2, arrowstyle='->', arrowsize=1.5)
    ax.scatter(x, y, s=marker_size, color='r')
    if show_z:
        ax.quiver(
            x,
            y,
            u,
            v,
            w,
            units='width',
            pivot='middle',
            cmap='seismic',
            #norm=colors.LogNorm(vmin=w.min(),vmax=w.max()),
        )
    else:
        ax.quiver(x, y, u, v, units='width', pivot='middle', cmap='seismic')
    if ylimit is not None:
        ax.set_ylim(ylimit[0], ylimit[1])
    #plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



def plot_ipyvolume(fname='./Spinhist.nc', size=5):
    hist=read_hist(fname)
    pos=hist["positions"]
    s=hist["spin"]
    ipv.figure()
    x,y,z=np.array(pos).T
    #vx, vy, vz=np.array(s).T
    
    vx=s[:,:,0]
    vy=s[:,:,1]
    vz=s[:,:,2]
    #ipv.quickscatter(x,y,z, size=1, marker='sphere')
    q=ipv.quiver(x, y, z, vx, vy, vz, size=size)
    ipv.animation_control(q, interval=10)
    #ipv.style.use(["dark","minimal"])
    ipv.show()

def plot_supercell(ham,
                   supercell_matrix=np.diag([30, 1, 1]),
                   plot_type='2d',
                   length=0.1,
                   marker_size=50,
                   show_z=False,
                   ylimit=None, 
                   temperature=0.0,
                   time_step=5e-4,
                   total_time=4,
                   pbc=[1,1,1],
                   magfield=False,
                   ):
    sc_ham = ham.make_supercell(supercell_matrix)
    def Hfunc(xcart):
        x, y, z = xcart
        if x< 0.2:
            return np.array([100,0,0.0])
        elif x> 0.8:
            return np.array([-100,0,0.0])
        else:
            return np.array([0,0,0.0])
    if magfield:
        sc_ham.set_spatial_external_field(Hfunc)
    sc_ham.s = np.random.rand(*sc_ham.s.shape) - 0.5
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(time_step=time_step, temperature=temperature, total_time=total_time, save_all_spin=False)

    mover.run(write_step=20)
    pos = np.dot(sc_ham.pos, sc_ham.cell)
    if plot_type == '2d':
        plot_2d_vector(
            pos, mover.s, show_z=show_z, length=length, ylimit=ylimit)
    elif plot_type == '3d':
        plot_3d_vector(pos, mover.s, length=length)
    elif plot_type == 'dynamic':
        plot_ipyvolume(size=length*20)



#exchange_1d()


def plot_spinwave(ham,
                  qnames=['$\Gamma$', 'X'],
                  qvectors=[(0, 0, 0), (0.5, 0, 0)]):
    fig = plt.figure()
    from ase.build import bulk
    from ase.dft.kpoints import get_special_points
    from ase.dft.kpoints import bandpath
    kpts, x, X = bandpath(qvectors, ham.cell, 300)
    qsolver = QSolver(hamiltonian=ham)
    evals, evecs = qsolver.solve_all(kpts, eigen_vectors=True)
    nbands = evals.shape[1]
    for i in range(nbands):
        plt.plot(x, evals[:, i] / 1.6e-21)
    plt.xlabel('Q-point (2$\pi$)')
    plt.ylabel('Energy (meV)')
    plt.xlim(x[0], x[-1])
    plt.xticks(X, qnames)
    for x in X:
        plt.axvline(x, linewidth=0.6, color='gray')
    plt.show()


def plot_M_vs_time(ham, supercell_matrix=np.eye(3), temperature=0):
    sc_ham = ham.make_supercell(supercell_matrix)
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(
        time_step=1e-5,
        temperature=temperature,
        total_time=1,
        save_all_spin=True)

    mover.run(write_step=10)

    hist = mover.get_hist()
    hspin = np.array(hist['spin'])
    time = np.array(hist['time'])
    tspin = np.array(hist['total_spin'])

    Ms = np.linalg.det(supercell_matrix)
    plt.figure()
    plt.plot(
        time, np.linalg.norm(tspin, axis=1) / Ms, label='total', color='black')
    plt.plot(time, tspin[:, 0] / Ms, label='x')
    plt.plot(time, tspin[:, 1] / Ms, label='y')
    plt.plot(time, tspin[:, 2] / Ms, label='z')
    plt.xlabel('time (s)')
    plt.ylabel('magnetic moment ($\mu_B$)')
    plt.legend()
    #plt.show()
    #avg_total_m = np.average((np.linalg.norm(tspin, axis=1)/Ms)[:])
    plt.show()


def plot_M_vs_T(ham, supercell_matrix=np.eye(3), Tlist=np.arange(0.0, 110,
                                                                 20)):
    Mlist = []
    sc_ham = ham.make_supercell(supercell_matrix)
    for temperature in Tlist:
        mover = SpinMover(hamiltonian=sc_ham)
        mover.set(
            time_step=2e-4,
            #damping_factor=0.1,
            temperature=temperature,
            total_time=1,
            save_all_spin=True)

        mover.run(write_step=10)

        hist = mover.get_hist()
        hspin = np.array(hist['spin'])
        time = np.array(hist['time'])
        tspin = np.array(hist['total_spin'])

        Ms = np.linalg.det(supercell_matrix)
        avg_total_m = np.average((np.linalg.norm(tspin, axis=1) / Ms)[300:])
        print("T: %s   M: %s" % (temperature, avg_total_m))
        Mlist.append(avg_total_m)

    plt.plot(Tlist, Mlist)
    plt.ylim(-0.01, 1.01)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average magnetization (Ms)')
    plt.show()


