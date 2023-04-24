from ase.io import read
from minimulti.lattice.lattice import Lattice
import matplotlib.pyplot as plt
import numpy as np

def read_phonon_model(ifc_file = 'data/BaTiO3.ifc', struct_file='data/BaTiO3.vasp'):
    atoms = read(struct_file)   # reference atomic structure
    calc = Lattice(ref_atoms=atoms)   # initialize Lattice model
    calc.read_ifc_file(fname=ifc_file) # IFC from ifcfile
    return calc

def plot_lattice_wannier(color='green', ind=0, emax=1.3, sigma=0.3, ifc_file = 'data/BaTiO3.ifc', struct_file='data/BaTiO3.vasp'):
    nk=3
    nR=3
    calc=read_phonon_model(ifc_file = ifc_file, struct_file=struct_file)
    lwf=calc.scdmk(nwann=3, ftype='unity', mu=emax, sigma=sigma, kpts=[nk,nk,nk], Rgrid=[nR,nR,nR],
    #lwf=calc.scdmk(nwann=3, ftype='unity', mu=emax, sigma=0, kpts=[3,3,3], Rgrid=[3,3,3],
    anchors={(.0,.0,.0):tuple(range(0,3))}, 
    #anchors=None,
    #selected_cols=[3,4,5],
                   )
    calc.plot_wann_band(lwf)
    plt.savefig('nk3.pdf')
    plt.show()
plot_lattice_wannier()
