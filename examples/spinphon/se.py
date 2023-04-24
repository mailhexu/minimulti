import numpy as np
from minimulti.electron.Hamiltonian import atoms_model
from minimulti.electron.basis2 import BasisSet, gen_basis_set, atoms_to_basis
from minimulti.electron.epc import EPC
from ase.atoms import Atoms
import matplotlib.pyplot as plt


def build_model(dx, U, spinat, Delta=1, nel=1, t=1, type='t2gt2g'):
    atoms = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
    bset = atoms_to_basis(atoms, basis_dict={'H': ('s', )}, nspin=2)
    mymodel = atoms_model(atoms=atoms, basis_set=bset, nspin=2)
    sc_matrix = np.diag([1, 1, 2])
    mymodel = mymodel.make_supercell(sc_matrix)
    # onsite energy (tune v to make electron localized on one site)
    # mymodel.set_onsite(v,0)
    # mymodel.set_onsite(v,1)
    # prepare electron-lattice term.
    C_vx = 0.0
    C_tx = 1.0
    dv = C_vx * dx*dx
    dt = C_tx * dx*dx

    # 1D hopping along z. Format: t, i, j, R.
    # spin up
    mymodel.set_hop(t - dt, 0, 2, [0, 0, 0])
    mymodel.set_hop(t - dt, 2, 0, [0, 0, 1])
    # spin down
    mymodel.set_hop(t + dt, 1, 3, [0, 0, 0])
    mymodel.set_hop(t + dt, 3, 1, [0, 0, 1])

    if type == 't2geg':
        #print(4 + spinat[1] * Delta)
        mymodel.set_onsite(3 - spinat[1] * Delta, 2, mode='reset')
        mymodel.set_onsite(3 - spinat[1] * Delta, 3, mode='reset')

    mymodel.set_initial_spin(spinat)
    mymodel.set_Hubbard_U(
        Utype='Kanamori',
        Hubbard_dict={'H': {
            'U': U,
            'J': 0,
            'L': 0
        }},
        DC_type='FLL-s')
    mymodel.set(nel=nel, mixing=0.3, tol_energy=1e-8, tol_rho=1e-8, sigma=0.1)
    return mymodel


def plot_epc(t=1,
             U=4.0,
             spinat=[1, -1],
             nel=1,
             C_vx=0.0,
             C_tx=0.0,
             dx=0.0,
             dtdx=1.0,
             order=1):
    mymodel = build_model(dx=dx, U=U, spinat=spinat, type='t2geg')
    mymodel.set_kmesh([1, 1, 30])
    # Scf solve
    mymodel.scf_solve(print_iter_info=False)
    efermi = mymodel.get_fermi_level()

    # Electron phonon coupling term.
    epc_nospin = EPC(norb=2)
    epc_nospin.add_term(R=(0, 0, 0), i=0, j=1, val=-dtdx)
    epc_nospin.add_term(R=(0, 0, 1), i=1, j=0, val=-dtdx)
    #epc_nospin.add_term(R=(0,0,0), i=0, j=1,val=dt)
    #epc_nospin.add_term(R=(0,0,0), i=0, j=1,val=dt)
    epc_spin = epc_nospin.to_spin_polarized()

    #shift=epc_spin.get_band_shift(kpts, evecs, mymodel.bset.get_positions())
    #print(shift)
    #print(epc_spin._epc)

    kpt_x = np.arange(0, 1.01, 0.001)
    kpts = [np.array([0, 0, 1]) * x for x in kpt_x]
    evalues, evecs = mymodel.solve_all(k_list=kpts)

    ax = epc_spin.plot_epc_fatband(
        kpts,
        evalues,
        evecs,
        kpt_x,
        order=order,
        X=[0, 1],
        xnames=['0', '1'],
        show=False,
        efermi=efermi,
        width=2)
    for i in range(mymodel._norb):
        ax.plot(kpt_x, evalues[i, :], color='green', linewidth=0.1)
    ax.set_xlabel('k-point')
    ax.set_ylabel('Energy')
    ax.axhline(efermi, linestyle='--', color='gray')

    plt.show()

#plot_epc(dx=0.3, spinat=[1,1],U=5)

def test_U_single_orb(t=1,
                      U=4.0,
                      spinat=[1, -1],
                      nel=1,
                      C_vx=0.0,
                      C_tx=0.0,
                      dx=0.0):
    mymodel = build_model(dx=dx, U=U, spinat=spinat, type='t2geg')
    mymodel.set_kmesh([1, 1, 30])

    # Scf solve
    mymodel.scf_solve(print_iter_info=False, convention=2)
    #return mymodel._int_energy+mymodel._DC_energy#+mymodel._Uband_energy
    #return mymodel._band_energy+mymodel._Uband_energy
    return mymodel._total_energy, mymodel._band_energy + mymodel._Uband_energy, mymodel._int_energy + mymodel._DC_energy


def lattice_energy(dx, K=0.1):
    return K * dx**2

#test()
def energy_vs_disp(spinat=[1, 1], label='FM', axes=None):
    dx_list = np.arange(-0.2, 0.201, 0.02)
    etot_list = []
    eelec_list = []
    etb_list = []
    eee_list = []
    for dx in dx_list:
        e_electron, etb, eee = test_U_single_orb(
            t=1, U=5, spinat=spinat, nel=1, dx=dx, C_tx=1, C_vx=0.0)
        e_lattice = lattice_energy(dx, K=0)
        e = e_electron + e_lattice
        etot_list.append(e)
        eelec_list.append(e_electron)
        etb_list.append(etb)
        eee_list.append(eee)
    mid_point = len(dx_list) // 2
    #es=np.array(e_list)
    #es=es-es[mid_point]
    axes.plot(dx_list, np.array(etot_list) - etot_list[mid_point], label=label)
    #axes[0].plot(dx_list,np.array(eelec_list)-eelec_list[mid_point], label=label)
    #axes[1].plot(dx_list,np.array(etb_list)-etb_list[mid_point])
    #axes[2].plot(dx_list,np.array(eee_list)-eee_list[mid_point])
    #axes[0].plot(dx_list,np.array(eelec_list), label=label)
    #axes[1].plot(dx_list,np.array(etb_list))
    #axes[2].plot(dx_list,np.array(eee_list))

def plot_energies():
    fig, axes = plt.subplots(1, sharex=True)
    energy_vs_disp(spinat=[1, 1], label='FM', axes=axes)
    energy_vs_disp(spinat=[1, -1], label='AFM', axes=axes)
    energy_vs_disp(spinat=[0.0, -0.0], label='NM', axes=axes)
    #plt.ylabel('E-E($\Delta x$=0)')
    #plt.xlabel('$\Delta$x')
    #axes[0].set_ylabel("$\Delta E$ (eV)")
    #axes[0].legend()
    #axes[1].set_ylabel("$\Delta E_{TB}$ (eV)")
    #axes[2].set_ylabel("$\Delta E_{ee}$ (eV)")
    #axes[2].set_xlabel("Amplitude")
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Amplitude')
    plt.ylabel('Energy (eV)')
    plt.subplots_adjust(hspace=0.01)
    plt.show()

plot_epc(dx=0.3, spinat=[1,1],U=5)
#plot_epc()
plot_energies()
