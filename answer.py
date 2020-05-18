import numpy as np
import sympy as sp
from math import sqrt, sin, pi


def matrix_a():
    n = 8
    A = np.matrix([[611, 196, -192, 407, -8, -52, -49, 29],
                   [196, 899, 113, -192, -71, -43, -8, -44],
                   [-192, 113, 899, 196, 61, 49, 8, 52],
                   [407, -192, 196, 611, 8, 44, 59, -23],
                   [-8, -71, 61, 8, 411, -599, 208, 208],
                   [-52, -43, 49, 44, -599, 411, 208, 208],
                   [-49, -8, 8, 59, 208, 208, 99, -911],
                   [29, -44, 52, -23, 208, 208, -911, 99]])

    e, v = np.linalg.eig(A)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))


def matrix_b():
    n = 10
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            B[i, j] = 1 / (i + j + 1)

    e, v = np.linalg.eig(B)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))


def matrix_c():
    n = 12
    C = np.matrix([[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   [11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   [10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   [9, 9, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   [8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1],
                   [7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1],
                   [6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1],
                   [5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1],
                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1],
                   [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1],
                   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    e, v = np.linalg.eig(C)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))


def matrix_d():
    n = 20
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = sqrt(2 / 21) * sin((i + 1) * (j + 1) * pi / 21)

    e, v = np.linalg.eig(D)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))


def matrix_e():
    n = 50
    E = np.zeros((n, n))
    for i in range(n):
        E[i, i] = 1
        E[i, n - 1] = 1
        for j in range(i):
            E[i, j] = -1

    e, v = np.linalg.eig(E)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))


def matrix_f():
    n = 11
    F = np.zeros((n, n))
    for i in range(1, n):
        F[i, i - 1] = 1
    for i in range(n):
        F[i, n - 1] = -1

    e, v = np.linalg.eig(F)
    e.sort()
    print('All:', e)
    print('Max:', max(abs(i) for i in e))
    print('Min:', min(abs(i) for i in e))

    x = sp.Symbol('x')
    f = x ** 11 + x ** 10 + x ** 9 + x ** 8 + x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1
    print(sp.solve(f))

    
if __name__ == '__main__':
    print('Matrix A')
    matrix_a()
    print()
    print('Matrix B')
    matrix_b()
    print()
    print('Matrix C')
    matrix_c()
    print()
    print('Matrix D')
    matrix_d()
    print()
    print('Matrix E')
    matrix_e()
    print()
    print('Matrix F')
    matrix_f()
    print()

