import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from minimulti.electron.builder.perovskite import perovskite_builder


def test(delta=1):
    model = perovskite_builder(
        A='La',
        B='Mn',
        O='O',
        orbs='eg',
        Hubbard_dict={'Mn': {
            'U': 0,
            'J': 0
        }},
    )
    sc_matrix = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]  # G-type
    nsc = int(np.linalg.det(sc_matrix))
    model = model.make_supercell(sc_matrix)
    #write('LaMnO3.cif', model.atoms, vasp5=True , sort=True)
    write('LaMnO3.cif', model.atoms)
    #model.set_initial_spin([1, 1]*nsc)
    # FM
    spin = [1, 1, 1, 1]
    # A-AFM
    #spin=[1,-1,1,-1]
    #G-AFM
    #spin=[1,-1,-1,1]
    # C-AFM
    #spin=[1,1,-1,-1]
    print(np.outer(spin, [-1, 1, -1, 1]).flatten())
    model.set_onsite(
        np.outer(spin, [-1, 1, -1, 1]).flatten() * delta, mode='reset')
    model.set(
        nel=1 * nsc, tol_energy=1e-6, tol_rho=1e-6, mixing=0.8, sigma=0.1)
    model.set_kmesh([6, 6, 4])
    #model.set_density(rho= n*n matrix, n is the number of orbs with spin)
    #model.load_density(fname="rho.npy")
    model.scf_solve()
    pdos=model.get_pdos(emin=-10, emax=10, nedos=100, sigma=0.2, fname='dos.txt')
    #model.save_density(fname="rho.npy")
    print("Charges of each orbital: ", model.get_charges())
    print("magnetic moment of each orbital:", model.get_magnetic_moments())
    ax = model.plot_band(supercell_matrix=sc_matrix, npoints=200)
    ax = model.plot_unfolded_band(sc_matrix)
    plt.show()


test()
