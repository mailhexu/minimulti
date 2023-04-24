import numpy as np
from ase.atoms import Atoms
from spindy.purepython.hamiltonian import Spin_Hamiltonian
from spindy.purepython.mover import spin_mover
import matplotlib.pyplot as plt


def test_cubic_curie(n_supercell, temperature=1):
    """
    Isolated spin in an applied field. field:10T, time step: 1fs.
    Total time: 100 ps, No Langevin term.
    """
    # make model
    atoms = Atoms(symbols="H", positions=[[0, 0, 0]], cell=[1, 1, 1])
    spin = np.array([[0, 0, 1]])

    ham = Spin_Hamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())

    ham.gilbert_damping=[1.0]

    Jval = 6e-21
    J = {
        (0, 0, (0, 0, 1)): Jval,
        (0, 0, (0, 0, -1)): Jval,
        (0, 0, (0, 1, 0)): Jval,
        (0, 0, (0, -1, 0)): Jval,
        (0, 0, (-1, 0, 0)): Jval,
        (0, 0, (1, 0, 0)): Jval,
    }

    ham.set_exchange_ijR(exchange_Jdict=J )

    #sc_ham=ham.make_supercell(np.eye(3)*n_supercell)
    sc_ham=ham.make_supercell(np.diag([3,3,3]))


    mover = spin_mover(hamiltonian=sc_ham)
    mover.set(
        time_step=0.01,
        #damping_factor=0.1,
        temperature=temperature,
        total_time=1,
        save_all_spin=False)

    #mover.run(write_step=10)
    mover.run_one_step()
    print(mover.current_s)
    mover.run_one_step()
    print(mover.current_s)
    return 

    hist = mover.get_hist()
    hspin = np.array(hist['spin'])
    time = np.array(hist['time'])
    tspin = np.array(hist['total_spin'])

    #plt.plot(time, hspin[:, 0, 0], label='x')
    #plt.plot(time, hspin[:, 1, 0], label='y')
    #plt.plot(time, hspin[:, 2, 0], label='z')
    #plt.legend()
    #plt.show()

    Ms=n_supercell**3
    plt.plot(time, np.linalg.norm(tspin, axis=1)/Ms, label='total', color='black')
    plt.plot(time, tspin[:, 0]/Ms, label='x')
    plt.plot(time, tspin[:, 1]/Ms, label='y')
    plt.plot(time, tspin[:, 2]/Ms, label='z')
    plt.legend()
    #plt.show()
    avg_total_m = np.average((np.linalg.norm(tspin, axis=1)/Ms)[100:])
    plt.savefig('fMt_n%s_T%s.png'%(n_supercell, temperature))
    plt.close()

    # Save M/Ms to file.
    print("T: %s \t M: %s \n"%(temperature, avg_total_m))
    with open('fresults_%s.txt'%n_supercell, 'a') as myfile:
        myfile.write("%s\t%s\n"%( temperature, avg_total_m))


test_cubic_curie(temperature=600, n_supercell=1)
#for n in [15]:
    #for T in [650]:
    #for T in np.arange(500,800,50):
#    for T in np.arange(500,800,50):
#        test_cubic_curie(temperature=T, n_supercell=n)

