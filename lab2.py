"""
@project: Numerical-and-Symbolic-Calculations
@author: jiahao Shen
@file: lab2.py
@ide: Visual Studio Code
"""
import numpy as np
from time import time
from eigen_values import *


def matrix_a():
    n = 8
    A = [611, 196, -192, 407, -8, -52, -49, 29,
         196, 899, 113, -192, -71, -43, -8, -44,
         -192, 113, 899, 196, 61, 49, 8, 52,
         407, -192, 196, 611, 8, 44, 59, -23,
         -8, -71, 61, 8, 411, -599, 208, 208,
         -52, -43, 49, 44, -599, 411, 208, 208,
         -49, -8, 8, 59, 208, 208, 99, -911,
         29, -44, 52, -23, 208, 208, -911, 99]

    return A, n


def matrix_b():
    n = 10
    B = [0 for _ in range(n * n)]

    for i in range(n):
        for j in range(n):
            B[i * n + j] = 1 / (i + j + 1)

    return B, n


def matrix_c():
    n = 12
    C = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
         11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
         10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
         9, 9, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1,
         8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1,
         7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1,
         6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1,
         5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    return C, n


def matrix_d():
    n = 20
    D = [0 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            D[i * n + j] = sqrt(2 / 21) * sin((i + 1) * (j + 1) * pi / 21)

    return D, n


def matrix_e():
    n = 50
    E = [0 for _ in range(n * n)]
    for i in range(n):
        E[i * n + i] = 1
        E[i * n + n - 1] = 1
        for j in range(i):
            E[i * n + j] = -1

    return E, n


def matrix_f():
    n = 11
    F = [0 for _ in range(n * n)]
    for i in range(1, n):
        F[i * n + i - 1] = 1
    for i in range(n):
        F[i * n + n - 1] = -1

    return F, n


def calculate_error(M, e, v, n):
    r = multiply(M, v, n, n, 1)
    for i in range(n):
        r[i] -= (e * v[i])

    print('误差: %.5E' % norm(r))


def main_1(matrix, n):
    print('-------------Power-------------')

    M = matrix[:]
    pld = [0]
    env = [0 for _ in range(n)]
    env[0] = 1

    t = time()
    power_eng(pld, env, M, n)
    t = time() - t

    print('最大特征值: %.5E' % pld[0])
    print('特征向量:', ['%.5E' % x for x in env])
    print('用时: %.5E ms'% (t * 1000))
    calculate_error(M, pld[0], env, n)
    print()

    print('-------------Inv Power-------------')

    M = matrix[:]
    pld = [0]
    env = [0 for _ in range(n)]
    env[0] = 1

    t = time()
    inv_power_eng(pld, env, M, n)
    t = time() - t

    print('最小特征值: %.5E' % pld[0])
    print('特征向量:', ['%.5E' % x for x in env])
    print('用时: %.5E ms' % (t * 1000))
    calculate_error(M, pld[0], env, n)
    print()

    print()


def main_2(matrix, n):
    print('-------------Jacobi-------------')

    M = matrix[:]
    env = [0 for _ in range(n)]

    t = time()
    jacobi_eng(env, M, n)
    t = time() - t
    env.sort()

    M = np.array(matrix).reshape(n, n)
    e, v = np.linalg.eig(M)
    e.sort()

    r = [0 for _ in range(n)]
    for i in range(n):
        r[i] = env[i] - e[i]

    print('特征向量:', ['%.5E' % x for x in env])
    print('误差: %.5E' % norm(r))
    print('用时: %.5E ms' % (t * 1000))
    print()


def main_3(matrix, n):
    print('-------------QR-------------')

    M = matrix[:]
    env = [0 for _ in range(n)]

    t = time()
    gauss_hessen(M, n)
    eig = Eigen()
    eig.qr_eng(env, M, n)
    t = time() - t
    env.sort(key=lambda x: (x.real, x.imag))

    M = np.array(matrix).reshape(n, n)
    e, v = np.linalg.eig(M)
    e.sort()

    r = [0 for _ in range(n)]
    for i in range(n):
        r[i] = env[i] - e[i]

    print('特征向量:', ['%.5E+%.5Ei' % (x.real, x.imag) for x in env])
    print('误差: %.5E' % norm(r))
    print('用时: %.5E ms' % (t * 1000))
    print()


def main_4(matrix, n):
    print('-------------QR-------------')

    M = matrix[:]
    env = [0 for _ in range(n)]

    t = time()
    gauss_hessen(M, n)
    eig = Eigen()
    eig.qr_eng(env, M, n)
    t = time() - t
    env.sort()

    print('方程解:', ['%.5E' % x for x in env])
    print('用时: %.5E ms' % (t * 1000))
    print()


if __name__ == '__main__':
    print('==================实验1==================')
    print('矩阵A')
    main_1(*matrix_a())
    print('矩阵B')
    main_1(*matrix_b())
    print('矩阵C')
    main_1(*matrix_c())
    print('矩阵D')
    main_1(*matrix_d())
 
    print('==================实验2==================')
    print('矩阵A')
    main_2(*matrix_a())
    print('矩阵B')
    main_2(*matrix_b())
    print('矩阵C')
    main_2(*matrix_c())
    print('矩阵D')
    main_2(*matrix_d())
 
    print('==================实验3==================')
    print('矩阵A')
    main_3(*matrix_a())
    print('矩阵B')
    main_3(*matrix_b())
    print('矩阵C')
    main_3(*matrix_c())
    print('矩阵D')
    main_3(*matrix_d())
    print('矩阵E')
    main_3(*matrix_e())

    print('==================实验4==================')
    print('11次方程组')
    main_4(*matrix_f())
