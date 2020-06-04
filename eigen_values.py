"""
@project: Numerical-and-Symbolic-Calculations
@author: jiahao Shen
@file: eigen_values.py
@ide: Visual Studio Code
"""
from utils import *
from random import random
from linear_equations import *
from cmath import sqrt as csqrt
from math import sqrt, atan, sin, cos, pi


def power_eng(pld, env, a, n):
    """使用幂法求矩阵的最大特征值及其特征向量
    @param pld: 特征值
    @param env: 特征向量
    @param a: 按行优先次序存放
    @param n: 矩阵维数
    @return: Boolean
    """
    for _ in range(1000):
        env_new = dot(a, env)

        l = norm(env_new)

        for i in range(n):
            env_new[i] /= l

        for i in range(n):
            env[i] = env_new[i]

        pld[0] = l

    return True


def inv_power_eng(pld, env, a, n):
    """使用反幂法求矩阵的最小特征值及其特征向量
    @param pld: 特征值
    @param env: 特征向量
    @param a: 按行优先次序存放
    @param n: 矩阵维数
    @return: Boolean
    """
    env_new = env[:]

    for _ in range(1000):
        A = [row[:] for row in a]
        pivot = list(range(n))

        lu(A, pivot, n)
        gauss(A, pivot, env_new, n)

        l = norm(env_new)

        for i in range(n):
            env_new[i] /= l

        for i in range(n):
            env[i] = env_new[i]

        pld[0] = 1 / l

    return True


def jacobi_eng(env, a, n):
    """使用Jacobi法求对称矩阵的全部特征值
    @param ev: 特征值
    @param a: 按行优先次序存放
    @param n: 矩阵维数
    @return: Boolean
    """
    for _ in range(1000):
        t, p, q = 0, -1, -1

        for i in range(n):
            for j in range(n):
                if i != j and abs(a[i][j]) > t:
                    t = abs(a[i][j])
                    p, q = i, j

        if abs(t) < 1e-9:
            return True

        if a[p][p] == a[q][q]:
            theta = pi / 4
        else:
            theta = atan(2 * a[p][q] / (a[p][p] - a[q][q])) / 2

        a_new = [row[:] for row in a]

        a_new[p][p] = a[p][p] * (cos(theta) ** 2) + a[q][q] * \
            (sin(theta) ** 2) + 2 * a[p][q] * cos(theta) * sin(theta)
        a_new[q][q] = a[p][p] * (sin(theta) ** 2) + a[q][q] * \
            (cos(theta) ** 2) - 2 * a[p][q] * cos(theta) * sin(theta)
        a_new[p][q] = (a[q][q] - a[p][p]) * sin(2 * theta) / \
            2 + a[p][q] * cos(2 * theta)
        a_new[q][p] = a_new[p][q]

        for i in range(n):
            if i != p and i != q:
                a_row = a_new[p][i]
                a_col = a_new[q][i]
                a_new[p][i] = a_row * cos(theta) + a_col * sin(theta)
                a_new[i][p] = a_new[p][i]
                a_new[q][i] = a_col * cos(theta) - a_row * sin(theta)
                a_new[i][q] = a_new[q][i]

        for i in range(n):
            for j in range(n):
                a[i][j] = a_new[i][j]

        for i in range(n):
            env[i] = a[i][i]

    return True


def gauss_hessen(a, n):
    """用Gauss相似变换将矩阵转换为Hessenberg矩阵
    @param a: 按行优先次序存放
    @param n: 矩阵维数
    @return: Boolean
    """
    for k in range(1, n - 1):
        i = k
        for j in range(k + 1, n):
            if abs(a[j][k - 1]) > abs(a[k][k - 1]):
                i = j

        t = a[i][k - 1]

        if abs(t) < 1e-9:
            return True

        if i != k:
            for j in range(k - 1, n):
                a[i][j], a[k][j] = a[k][j], a[i][j]
            for j in range(n):
                a[j][i], a[j][k] = a[j][k], a[j][i]
        for i in range(k + 1, n):
            m = a[i][k - 1] / t
            a[i][k - 1] = 0
            for j in range(k, n):
                a[i][j] -= m * a[k][j]
            for j in range(n):
                a[j][k] += m * a[j][i]

    return False


class Eigen(object):

    def __init__(self):
        pass

    def qr_aux(self, i, j):
        """递归对子矩阵进行QR迭代
        @param i: 子矩阵的左边界
        @param j: 子矩阵的右边界
        @return: Boolean
        """
        if j - i == 1:
            self.en[i] = self.a[i][i]
            return

        if j - i == 2:
            a = self.a[i][i]
            b = self.a[i][i + 1]
            c = self.a[i + 1][i]
            d = self.a[i + 1][i + 1]
            delta = (a + d) ** 2 - 4 * (a * d - b * c)

            if delta >= 0:
                self.en[i] = (a + d + sqrt(delta)) / 2
                self.en[i + 1] = (a + d - sqrt(delta)) / 2
            else:
                self.en[i] = (a + d + csqrt(delta)) / 2
                self.en[i + 1] = (a + d - csqrt(delta)) / 2
            return

        A = [[0 for _ in range(j - i)] for _ in range(j - i)]
        for x in range(j - i):
            for y in range(j - i):
                A[x][y] = self.a[i + x][i + y]

        for _ in range(50):
            for k in range(1, j - i):
                if abs(A[k][k - 1]) < 1e-9:
                    for x in range(j - i):
                        for y in range(j - i):
                            self.a[i + x][i + y] = A[x][y]

                    self.qr_aux(i, i + k)
                    self.qr_aux(i + k, j)
                    return False

            Q, R = self.qr_solve(A, j - i)
            A = multiply(R, Q)

        for k in range(i, j):
            self.en[k] = A[k - i][k - i]

        return False

    def qr_eng(self, result, h, m):
        """驱动函数
        @param result: 特征值
        @param h: 上Hessenberg矩阵
        @param m: 矩阵维数
        @return: 
        """
        self.a = h
        self.n = m
        self.en = result

        self.qr_aux(0, self.n)

    def qr_solve(self, A, n):
        """对子矩阵进行QR分解
        @param A: 按行优先次序存放
        @param n: 矩阵维数
        @return: Q, R
        """
        Q = identity(n)
        R = [row[:] for row in A]

        for i in range(n - 1):
            x = []
            for j in range(i, n):
                x.append(R[j][i])

            u = x[:]
            u[0] = u[0] - norm(x)

            v = []
            l = norm(u)
            for j in range(len(u)):
                v.append(u[j] / l)

            h = identity(n)
            v2 = outer(v, v)
            for j in range(len(v)):
                for k in range(len(v)):
                    h[j + i][k + i] -= 2 * v2[j][k]

            R = multiply(h, R)
            Q = multiply(Q, h)

        return Q, R


def test_power_eng():
    print('==========Test Power Eng==========')
    print('----------Case 1----------')
    n = 3
    a = [[-4, 14, 0],
         [-5, 13, 0],
         [-1, 0, 2]]
    pld = [0]
    env = [random() for _ in range(n)]

    power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [6.000005847178563]
    # [0.7974005138049264, 0.5695721214871647, -0.19934898797899614]
    print()

    print('----------Case 2----------')
    n = 4
    a = [[2, 1, 0, 0],
         [1, 2, 1, 0],
         [0, 1, 2, 1],
         [0, 0, 1, 2]]
    pld = [0]
    env = [random() for _ in range(n)]
    power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [3.618033988739841]
    # [0.37174597998559483, 0.6014996852702518, 0.6015022247378224, 0.3717500889304373]
    print()

    print('----------Case 3----------')
    n = 3
    a = [[2, -1, 0],
         [0, 2, -1],
         [0, -1, 2]]
    pld = [0]
    env = [random() for _ in range(n)]
    power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [3.000002060281634]
    # 0.5773478901706934, -0.5773514586954156, 0.5773514586954162]
    print()

    print()


def test_inv_power_eng():
    print('==========Test Inv Power Eng==========')
    print('----------Case 1----------')
    n = 3
    a = [[2, -1, 0],
         [-1, 2, -1],
         [0, -1, 2]]
    pld = [0]
    env = [random() for _ in range(n)]
    inv_power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [0.5857864376272454]
    # [0.4999992031718274, 0.707106780725981, 0.5000007974782413]
    print()

    print('----------Case 2----------')
    n = 3
    a = [[-4, 14, 0],
         [-5, 13, 0],
         [-1, 0, 2]]
    pld = [0]
    env = [random() for _ in range(n)]
    inv_power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [1.9999985756069438]
    # [-2.1365958615155277e-06, -1.0682979307444387e-06, -0.9999999999971467]
    print()

    print('----------Case 3----------')
    n = 3
    a = [[-1, 1, 0],
         [-4, 3, 0],
         [1, 0, 2]]
    pld = [0]
    env = [random() for _ in range(n)]
    inv_power_eng(pld, env, a, n)
    print(pld)
    print(env)
    # [0.9990003761062305]
    # [0.408452320847682, 0.8164965469326176, -0.4080442260849357]
    print()

    print()


def test_jacobi_eng():
    print('==========Test Jacobi Eng==========')
    print('----------Case 1----------')
    n = 3
    a = [[4, 2, 2],
         [2, 5, 1],
         [2, 1, 6]]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [2.12592446854474, 8.387619058475414, 4.486456472979847]
    print()

    print('----------Case 2----------')
    n = 4
    a = [[1, 1, 1, 1],
         [1, 2, 3, 4],
         [1, 3, 6, 10],
         [1, 4, 10, 20]]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.4538345500256738, 2.2034461676473147, 0.038016015229141276, 26.304703267097885]

    print('----------Case 3----------')
    n = 4
    a = [[4, -30, 60, -35],
         [-30, 300, -675, 420],
         [60, -675, 1620, -1050],
         [-35, 420, -1050, 700]]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.16664286117190066, 37.10149136512766, 2585.253810928919, 1.4780548447781765]
    print()

    print('----------Case 4----------')
    n = 3
    a = [[3.5, -6, 5],
         [-6, 8.5, -9],
         [5, -9, 8.5]]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.4659302062458505, -0.9340137468008786, 20.968083540555032]
    print()

    print('----------Case 5----------')
    n = 3
    a = [[6, 2, 4],
         [2, 3, 2],
         [4, 2, 6]]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [11.0, 2.0, 1.9999999999999998]
    print()


def test_gauss_hessen():
    print('==========Test Gauss Hessenberg==========')
    print('----------Case 1----------')
    n = 4
    A = [[9, 18, 9, -27],
         [18, 45, 0, -45],
         [9, 0, 126, 9],
         [-27, -45, 9, 135]]

    gauss_hessen(A, n)
    output(A)
    # 9.00000E+00     -4.20000E+01    2.37273E+01     9.00000E+00
    # -2.70000E+01    1.62000E+02     -3.92727E+01    9.00000E+00
    # 0.00000E+00     3.30000E+01     1.88182E+01     6.00000E+00
    # 0.00000E+00     0.00000E+00     5.51157E+01     1.25182E+02
    print()

    print('----------Case 2----------')
    n = 4
    A = [[54, 40, 10, 76],
         [47, 20, 94, 49],
         [26, 80, 94, 70],
         [3, 92, 83, 45]]

    gauss_hessen(A, n)
    output(A)
    # 5.40000E+01     5.03830E+01     8.29790E+01     1.00000E+01
    # 4.70000E+01     7.51277E+01     1.14602E+02     9.40000E+01
    # 0.00000E+00     1.35992E+02     9.56103E+01     7.70000E+01
    # 0.00000E+00     0.00000E+00     5.47923E+00     -1.17380E+01
    print()

    print('----------Case 3----------')
    n = 3
    A = [[-149, -50, -154],
         [537, 180, 546],
         [-27, -9, -25]]

    gauss_hessen(A, n)
    output(A)
    # -1.49000E+02    -4.22570E+01    -1.54000E+02
    # 5.37000E+02     1.52547E+02     5.46000E+02
    # 0.00000E+00     -7.30314E-02    2.45251E+00
    print()

    print('----------Case 4----------')
    n = 3
    A = [[1, 3, 4],
         [3, 2, 1],
         [4, 1, 3]]

    gauss_hessen(A, n)
    output(A)
    # 1.00000E+00     6.25000E+00     3.00000E+00
    # 4.00000E+00     3.75000E+00     1.00000E+00
    # 0.00000E+00     -3.12500E-01    1.25000E+00
    print()

    print('----------Case 5----------')
    n = 4
    A = [[8, 6, 10, 10],
         [9, 1, 10, 5],
         [1, 3, 1, 8],
         [10, 6, 10, 1]]

    gauss_hessen(A, n)
    output(A)
    # 8.00000E+00     1.64000E+01     1.01431E+01     6.00000E+00
    # 1.00000E+01     7.40000E+00     1.01431E+01     6.00000E+00
    # 0.00000E+00     1.00600E+01     5.72565E-02     2.40000E+00
    # 0.00000E+00     0.00000E+00     8.93664E-01     -4.45726E+00
    print()


def test_hessen_qr():
    print('==========Test QR Iteration==========')
    print('----------Case 1----------')
    n = 3
    A = [[5, -3, 2],
         [6, -4, 4],
         [4, -4, 5]]

    gauss_hessen(A, n)
    output(A)

    env = [0 for _ in range(n)]
    eig = Eigen()
    eig.qr_eng(env, A, n)
    print(env)
    # [3.0000000000000018, 1.9999999999999991, 1.0000000000000009]
    print()

    print('----------Case 2----------')
    n = 5
    A = [[5, -5/3, 2, 4, 5],
         [6, -4/3, 4, 5, 6],
         [0, 2/9, 7/3, 6, 7],
         [0, 0, 0, 3, -2],
         [0, 0, 0, 4, -1]]

    env = [0 for _ in range(n)]
    eig = Eigen()
    eig.qr_eng(env, A, n)
    print(env)
    # [2.999999999999992, 2.000000001465689, 0.9999999985343235, (1+2j), (1-2j)]
    print()

    print('----------Case 3----------')
    n = 2
    A = [[0, -1],
         [1, 0]]

    env = [0 for _ in range(n)]
    eig = Eigen()
    eig.qr_eng(env, A, n)
    print(env)
    # [1j, -1j]
    print()

    print()


if __name__ == '__main__':
    test_power_eng()
    test_inv_power_eng()
    test_jacobi_eng()
    test_gauss_hessen()
    test_hessen_qr()
