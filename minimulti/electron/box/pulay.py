import numpy as np
from .dummymixer import DummyMixer


class PulayMixer(DummyMixer):
    """ A class for Pulay mixing, as described in
        https://wiki.fysik.dtu.dk/gpaw/documentation/densitymix/densitymix.html#densitymix """
    def __init__(self,
                 mixing_constant=0.3,
                 memory=8,
                 convergence=1e-3,
                 chop=None,
                 simple=False):
        DummyMixer.__init__(self, mixing_constant, convergence)
        self.name = 'Pulay'
        self.memory = memory
        self.initialized = False
        self.A = np.zeros((self.memory, self.memory))
        self.alfa = np.zeros(self.memory)
        self.chop = chop
        self.simple = simple

    def _initialize(self, xi):
        """ Allocate fixed-sized arrays. """
        self.R = np.zeros((self.memory, len(xi)))
        self.rho = np.zeros((self.memory, len(xi)))
        self.initialized = True

    def __call__(self, xi, yi):
        simple = self.simple
        if self.it == 0:
            self._initialize(xi)
        self.it += 1
        lim = min(self.it, self.memory)
        self.rho[1:lim] = self.rho[0:lim - 1]
        self.rho[0] = xi
        self.R[1:lim] = self.R[0:lim - 1]
        self.R[0] = yi - xi
        # todo : could use history.
        np.einsum('im, jm -> ij',
                  self.R[:lim, :],
                  self.R[:lim, :],
                  out=self.A[:lim, :lim])
        try:
            inv_A = np.linalg.inv(self.A[:lim, :lim])
        except:
            # problem in taking the inverse of the matrix A,
            # use simple mixing in this step.
            simple = True
        if simple or self.it < 20:
            xb = (1 - self.beta) * xi + self.beta * yi
        else:
            down = np.sum(inv_A[:lim, :lim])
            for i in range(lim):
                up = np.sum(inv_A[:lim, i])
                self.alfa[i] = up / down
            xb = np.zeros_like(xi)

            for i in range(lim):
                xb += self.alfa[i] * (self.rho[i] + self.beta * self.R[i])

        # The input must not change more than chop for all
        maxdev = max(abs(xb - xi))
        if self.chop != None and maxdev > self.chop:
            xb = xi + self.chop / maxdev * (xb - xi)

        fmax = max(abs(self.R[0]))
        self.fmax.append(fmax)
        if self.it > 3 and np.all(np.array(self.fmax[-3:]) < self.convergence):
            return True, xb
        else:
            return False, xb
