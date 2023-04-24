import numpy as np
from ase.atoms import Atoms
from minimulti.spin import SpinHamiltonian
from minimulti.spin import SpinMover
import matplotlib.pyplot as plt


def test_single_spin():
    """
    Isolated spin in an applied field. field:10T, time step: 1fs.
    Total time: 100 ps, No Langevin term.
    """
    # make model
    atoms = Atoms(symbols="H", positions=[[0, 0, 0]], cell=[1, 1, 1])
    spin = np.array([[0, 1, 0]], dtype=float)

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.set(gilbert_damping=[0.0])
    ham.set_external_hfield([0, 0.0, 10.0])

    mover = SpinMover(hamiltonian=ham, s=spin, write_hist=True)
    mover.set(
        time_step=0.01,
        temperature=0.0,
        total_time=10,
        save_all_spin=True)

    mover.run(write_step=1, method='DM')

    hist = mover.get_hist()

    hspin = np.array(hist['spin'])
    time = np.array(hist['time'])

    plt.plot(time, hspin[:, 0, 0], label='x')
    plt.plot(time, hspin[:, 0, 1], label='y')
    plt.plot(time, hspin[:, 0, 2], label='z')
    plt.legend()
    plt.show()


test_single_spin()
