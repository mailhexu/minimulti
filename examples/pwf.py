from ase.io import read
from minimulti.lattice.lattice import Lattice
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_phonon_model(ifc_file = 'data/BaTiO3.ifc', struct_file='data/BaTiO3.vasp'):
    atoms = read(struct_file)   # reference atomic structure
    calc = Lattice(ref_atoms=atoms)   # initialize Lattice model
    calc.read_ifc_file(fname=ifc_file) # IFC from ifcfile
    return calc

def plot_lattice_wannier(color='green', ind=0, emax=0.1, sigma=0.1, ifc_file = 'data/BaTiO3.ifc', struct_file='data/BaTiO3.vasp'):
    nk=7
    nR=7
    calc=read_phonon_model(ifc_file = ifc_file, struct_file=struct_file)
    lwf=calc.projwann(nwann=3, ftype='unity', mu=emax, sigma=0, kpts=[nk,nk,nk], Rgrid=[nR,nR,nR],
    anchors={(.0,.0,.0):tuple(range(0,3))}, 
                   )
    calc.plot_wann_band(lwf)
    #print(lwf.wannR)
    #print(lwf.Rlist)
    #for iR, R in enumerate(lwf.Rlist):
    #    print(R)
    #    print(np.real(lwf.wannR[iR]))
    plt.savefig('pwf.pdf')
    plt.show()
plot_lattice_wannier()
