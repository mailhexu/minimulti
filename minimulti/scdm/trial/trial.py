import numpy as np
from scipy.linalg import qr, eigh, svd

from minimulti.math.linalg import Lowdin


class LowdinOrth():
    def __init__(self, S):
        self.S = S
        U, _, VT = svd(S)
        #self.Smh=U@VT
        self.Smh = Lowdin(S)

    def orth_psi(self, psi):
        return  self.Smh.T.conj()@psi @self.Smh

    def orth_H(self, H):
        return self.Smh.T.conj() @ H @ self.Smh


def test_LowdinOrth(S, H):
    O=LowdinOrth(S)
    Ho=O.orth_H(H)
    e, p=eigh(Ho)
    print("EigHo evals",e)
    print("EigHo evecs",p)

def non_ortho_hamiltonian():
    n = 3
    H = np.random.random((n, n))
    H = (H + H.T.conj()) / 2.0

    S = np.eye(3)
    S[0, 1] = 0.3
    S[0, 2] = 0.2
    S[1, 2] = 0.27
    S = (S + S.T.conj()) / 2.0

    evals, evecs = eigh(H, b=S)

    test_LowdinOrth(S, H)
    O=LowdinOrth(S)
    print(f"{O.orth_psi(evecs)}")


    CHC = evecs.T @ H @ evecs
    P = evecs.T.conj() @ evecs
    print("Evals:", evals)
    print(sum(evals))
    print(f"H: {H}")
    print("H@P:", np.trace(H @ P))
    print("E@P", np.diag(evals) @ P)
    print("CHC:", CHC)


def main():
    non_ortho_hamiltonian()


main()
