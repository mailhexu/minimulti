#!/usr/bin/env python
import math
import numpy as np
import scipy.linalg as linalg

class QSolver(object):
    def __init__(self, hamiltonian):
        self.ham = hamiltonian
        self.nspin = self.ham.nspin
        M = linalg.norm(self.ham.spinat, axis=1)
        self.M_mat=np.kron(np.sqrt(np.einsum('i,j->ij', M, M)), np.ones((3,3)))
        self.Eref=None

    def get_Eref(self):
        mat = np.zeros((3 * self.nspin, 3 * self.nspin), dtype=complex)
        for key, val in self.ham.get_total_hessian_ijR().items():
            i, j, R = key
            mat[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= val
        mat=mat*4.0/self.M_mat
        SZ=np.zeros(self.nspin*3,dtype=float)
        SZ[2::3]=1.0
        SZ/=np.linalg.norm(SZ)
        self.Eref=np.dot(SZ, np.dot(mat, SZ)).real
        return self.Eref

    def solve_k(self, kpt, eigen_vectors=True, Jq=False):
        if self.Eref is None and not Jq:
            self.get_Eref()
        mat = np.zeros((3 * self.nspin, 3 * self.nspin), dtype=complex)
        for key, val in self.ham.get_total_hessian_ijR().items():
            i, j, R = key
            mat[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= val * np.exp(
               2.0j * math.pi * np.dot(kpt, R))
        if not Jq:
            mat=mat*4.0/self.M_mat
        if eigen_vectors:
            evals, evecs = linalg.eigh(mat)
            evals=np.real(evals)
            if not Jq:
                evals-=self.Eref
            return evals, evecs
        else:
            evals = np.linalg.eigvalsh(mat)
            if not Jq:
                evals-=self.Eref
            return evals

    def solve_all(self, kpts, eigen_vectors=True, Jq=False):
        eval_list = []
        evec_list = []
        if eigen_vectors:
            for kpt in kpts:
                evals, evecs = self.solve_k(kpt, eigen_vectors=True, Jq=Jq)
                eval_list.append(evals)
                evec_list.append(evecs)
            return np.array(eval_list), np.array(evec_list)
        else:
            for kpt in kpts:
                evals = self.solve_k(kpt, eigen_vectors=False, Jq=Jq)
                eval_list.append(evals)
            return np.array(eval_list)
