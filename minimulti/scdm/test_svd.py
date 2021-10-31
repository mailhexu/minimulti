from scipy.linalg import svd
from numpy.random import random

def test():
    A=random((3,4))
    print(A)
    U, S, VT = svd(A, full_matrices=False)
    print(U, S, VT)
    print(U.shape, S.shape, VT.shape)
    print(U@VT)

test()
