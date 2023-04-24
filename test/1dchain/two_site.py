import numpy as np
from  minimulti.electron.Hamiltonian import etb_model, atoms_model
from minimulti.electron.basis2 import Basis_Set, gen_basis_set, atoms_to_basis
from minimulti.electron.epc import EPC
from ase.atoms import Atoms
import matplotlib.pyplot as plt


def test_U_single_orb(t=1, U=9.9, spinat=[1,-1], nel=2, v=[0.0,0.0], dt=1):
    # generate structure.
    atoms=Atoms(symbols='H2', positions=[(0,0,0),(0,0,1)], cell=[1,1,2])

    # generate basis set
    bset= atoms_to_basis(atoms, basis_dict={'H': ('s',)}, nspin=2)

    # generate model. 
    # dim_k: dimension of k space
    # dim_r: dimension of real space.
    # lat: lattice parameter
    # orb: position of orbitals. Here orbs are
    #  0: H (0,0,0) s orbital spin up
    #  1: H (0,0,0) s orbital spin down
    #  2: H (0,0,1) s orbital spin up
    #  3: H (0,0,1) s orbital spin down
    #mymodel=etb_model(dim_k=3, dim_r=3, lat=np.diag([1,1,2]), orb=[[0,0,0], [0,0,0], [0,0,1],[0,0,1]], nspin=2)
    mymodel=atoms_model(atoms=atoms, basis_set=bset, nspin=2)

    # initial magnetic moment.
    mymodel.set_initial_spin(spinat)

    # onsite energy (tune v to make electron localized on one site)
    #mymodel.set_onsite(v,0)
    #mymodel.set_onsite(v,1)
    #mymodel.set_onsite(v,2)
    #mymodel.set_onsite(v,3)


    # 1D hopping along z. Format: t, i, j, R.
    mymodel.set_hop(t+dt, 0,2, [0,0,0])
    mymodel.set_hop(t+dt, 1,3, [0,0,0])
    mymodel.set_hop(t-dt, 2,0, [0,0,1])
    mymodel.set_hop(t-dt, 3,1, [0,0,1])
    #mymodel.set_hop(t, 0,2, [0,0,0])
    #mymodel.set_hop(t, 1,3, [0,0,0])
    #mymodel.set_hop(t, 2,0, [0,0,1])
    #mymodel.set_hop(t, 3,1, [0,0,1])

    mymodel.set(nel=nel, mixing=0.5, tol_energy=1e-8, tol_rho=1e-8)

    print(mymodel._hoppings)

    mymodel.set_Hubbard_U(Utype='Dudarev', Hubbard_dict={'H':{'U':U, 'J':0}})

    # K mesh.
    mymodel.set_kmesh([1,1,30])

    # Scf solve
    mymodel.scf_solve()

    efermi=mymodel.get_fermi_level()

    kpt_x=np.arange(0, 1.01,0.01)
    kpts=[np.array([0,0,1]) * x for x in kpt_x]
    evalues, evecs=mymodel.solve_all(k_list=kpts)

    epc_nospin=EPC(norb=2)
    epc_nospin.add_term(R=(0,0,0), i=0, j=1,val=dt)
    epc_nospin.add_term(R=(0,0,1), i=1, j=0,val=-dt)
    #epc_nospin.add_term(R=(0,0,0), i=0, j=1,val=dt)
    #epc_nospin.add_term(R=(0,0,0), i=0, j=1,val=dt)
    epc_spin=epc_nospin.to_spin_polarized()

    shift=epc_spin.get_band_shift(kpts, evecs, mymodel.bset.get_positions())
    print(shift)
    print(epc_spin._epc)
    epc_spin.plot_epc_fatband(kpts, evalues, evecs, bset.get_positions(), kpt_x, X=[0,1], xnames=['G','Pi'], show=True, efermi=efermi, width=100)
    for i in range(mymodel._norb):
        plt.plot(kpt_x, evalues[i,:])
    plt.xlabel('k-point')
    plt.ylabel('Energy')
    plt.axhline(efermi, linestyle='--', color='gray')

#test()
test_U_single_orb(t=1, U=5.000, spinat=[1,1], nel=2, v=0, dt=0.05)
#test_U_single_orb(t=1, U=4.059, spinat=[1,1], nel=2, v=0, dt=0.1)
#test_U_single_orb(t=1, U=4.059, spinat=[1,1], nel=2, v=0, dt=0.2)
plt.show()
