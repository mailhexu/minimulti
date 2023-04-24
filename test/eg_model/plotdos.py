import numpy as np
import matplotlib.pyplot as plt

def plot_dos():
    dos=np.loadtxt('dos.txt') 
    ndos=dos.shape[1]-1
    for i in range(1, ndos+1):
        plt.plot(dos[:,0],dos[:,i], label='%s'%i)
    plt.legend()
    plt.show()

plot_dos()
