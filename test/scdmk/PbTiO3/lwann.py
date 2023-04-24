from ase.io import read
from minimulti.lattice.lattice import Lattice
import matplotlib.pyplot as plt

def read_BTO_model(ifcfile = 'PTO.ifc'):
    atoms = read('POSCAR')   # reference atomic structure
    calc = Lattice(ref_atoms=atoms)   # initialize Lattice model
    calc.read_ifc_file(fname=ifcfile) # IFC from ifcfile
    return calc

def plot_phonon(color='green', ind=0):
    calc=read_BTO_model(ifcfile = 'PTO.ifc')
    ax = calc.plot_phonon_band(color=color, ax=None)
    #ax = calc.scdmk(nwann=3, ftype='Fermi', mu=-100, sigma=10, ax=ax, anchor_only=False,anchors={(.0,.0,.0):(0,1,2)}, scols=(9,10,11))
    #ax = calc.scdmk(nwann=3, ftype='Fermi', mu=800, sigma=2, ax=ax, anchor_only=True,anchors={(.0,.0,.0):(0,1,2)})
    ax = calc.scdmk(nwann=3, ftype='Fermi', mu=3000, sigma=200, ax=ax, anchor_only=False,anchors={(.0,.0,.0):(0,1,2)}, scols=[3*ind, 3*ind+1, 3*ind+2])
    plt.savefig('PTO_branch_%s.png'%ind, dpi=200)
    plt.show()
for i in range(5):
    plot_phonon(ind=i)
