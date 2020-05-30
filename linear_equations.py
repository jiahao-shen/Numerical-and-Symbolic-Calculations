"""
@project: Numerical-and-Symbolic-Calculations
@author: jiahao Shen
@file: linear_equations.py
@ide: Visual Studio Code
"""
from utils import *


def lu(a, pivot, n):
    """使用高斯列选主元消去法将矩阵进行LU分解
    @param a: 按行优先次序存放
    @param pivot: 输出参数, 存放主元的位置排列
    @param n: 矩阵维数
    @return: Boolean
    """
    if len(a) != n * n:
        return True

    for i in range(n - 1):
        max_t = abs(a[i * n + i])
        t = i

        for j in range(i + 1, n):
            if abs(a[j * n + i]) > max_t:
                max_t = abs(a[j * n + i])
                t = j

        if abs(max_t) < 1e-9:
            return True

        for j in range(n):
            a[i * n + j], a[t * n + j] = a[t * n + j], a[i * n + j]

        pivot[i], pivot[t] = pivot[t], pivot[i]

        for j in range(i + 1, n):
            a[j * n + i] = a[j * n + i] / a[i * n + i]

            for k in range(i + 1, n):
                a[j * n + k] = a[j * n + k] - a[i * n + k] * a[j * n + i]

    return False


def gauss(lu, p, b, n):
    """使用高斯列选主元消去法求解线性代数方程组
    @param lu: 按行优先次序存放的LU分解
    @param p: LU分解的主元排列
    @param b: 线性方程组Ax=b的右端向量
    @param n: 矩阵维数
    @return: Boolean
    """
    temp = b[:]

    for i in range(n):
        b[i] = temp[p[i]]

    for i in range(n):
        for j in range(i):
            b[i] = b[i] - lu[i * n + j] * b[j]

    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            b[i] = b[i] - lu[i * n + j] * b[j]
        b[i] = b[i] / lu[i * n + i]

    return False


def qr(a, d, n):
    """矩阵的QR分解
    @param a: 按行优先次序存放
    @param d: QR分解后的上三角矩阵的对角线元素
    @param n: 矩阵维数
    @return: Boolean
    """
    temp = [0 for _ in range(n)]

    for i in range(n - 1):
        m = 0
        for j in range(i, n):
            m += (a[j * n + i] ** 2)
        if a[i * n + i] > 0:
            m = -sqrt(m)
        else:
            m = sqrt(m)

        t = 0
        d[i] = m
        a[i * n + i] -= m

        for j in range(i, n):
            t += (a[j * n + i] ** 2)
        t = sqrt(t)

        for j in range(i, n):
            a[j * n + i] /= t

        for j in range(i + 1, n):
            for k in range(i, n):
                t = 0
                for l in range(i, n):
                    t += (a[k * n + i] * a[l * n + i] * a[l * n + j])
                temp[k] = a[k * n + j] - 2 * t

            for k in range(i, n):
                a[k * n + j] = temp[k]

    d[n - 1] = a[(n - 1) * n + n - 1]


def householder(qr, d, b, n):
    """使用Householder变换法求解线性代数方程组
    @param qr: QR分解后的矩阵
    @param d: QR分解后的上三角矩阵的对角线元素
    @param b: 线性方程组Ax=b的右端向量
    @param n: 矩阵维数
    @return: Boolean
    """
    temp = [0 for _ in range(n)]

    for i in range(n - 1):
        for j in range(i, n):
            t = 0
            for k in range(i, n):
                t += (qr[k * n + i] * qr[j * n + i] * b[k])
            temp[j] = b[j] - 2 * t
        for j in range(i, n):
            b[j] = temp[j]

    for i in reversed(range(0, n)):
        for j in reversed(range(i + 1, n)):
            b[i] -= (b[j] * qr[i * n + j])
        b[i] /= d[i]

    return False


def test_lu():
    print('==========Test LU==========')

    print('----------Case 1----------')
    n = 4
    A = [1, 2, 3.75, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    pivot = list(range(4))

    lu(A, pivot, n)
    output(A, n, n)
    # 3.00000E+00     7.00000E+00     1.00000E+00     0.00000E+00
    # 3.33333E-01     2.66667E+00     4.66667E+00     -5.00000E+00
    # 3.33333E-01     -1.25000E-01    4.00000E+00     -6.25000E-01
    # 6.66667E-01     5.00000E-01     -7.50000E-01    4.03125E+00
    print(pivot)
    # [1, 3, 0, 2]
    print()

    print()


def test_gauss():
    print('==========Test Gauss==========')

    print('----------Case 1----------')
    n = 4
    A = [1, 2, 0, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    b = [3, 11, 10, 6]
    pivot = list(range(4))

    lu(A, pivot, n)
    gauss(A, pivot, b, n)
    print(b)
    # [1.0000000000000009, 0.9999999999999993, 1.000000000000001, 1.0000000000000009]
    print()

    print()


def test_qr():
    print('==========Test QR==========')

    print('----------Case 1----------')
    n = 3
    A = [12, -51, 4,
         6, 167, -68,
         -4, 24, -41]
    d = [0 for _ in range(n)]

    qr(A, d, n)
    output(A, n, n)
    # 9.63624E-01     -2.10000E+01    1.40000E+01
    # 2.22375E-01     9.98460E-01     7.00000E+01
    # -1.48250E-01    5.54700E-02     -3.50000E+01
    print(d)
    # [-14.0, -175.0, -35.0]
    print()

    print('----------Case 2----------')
    n = 4
    A = [5, -2, -5, -1,
         1, 0, -3, 2,
         0, 2, 2, -3,
         0, 0, 1, -2]
    d = [0 for _ in range(n)]

    qr(A, d, n)
    output(A, n, n)
    # 9.95133E-01     1.96116E+00     5.49125E+00     5.88348E-01
    # 9.85376E-02     7.72156E-01     -1.58519E+00    2.52875E+00
    # 0.00000E+00     6.35433E-01     9.79199E-01     3.26718E+00
    # 0.00000E+00     0.00000E+00     2.02900E-01     -7.64719E-01
    print(d)
    # [-5.0990195135927845, -2.0380986614602725, -2.516611478423583, -0.7647191129018727]
    print()

    print('----------Case 3----------')
    n = 3
    A = [0, 3, 1,
         0, 4, -2,
         2, 1, 1]
    d = [0 for _ in range(n)]

    qr(A, d, n)
    output(A, n, n)
    # -7.07107E-01    1.00000E+00     1.00000E+00
    # 0.00000E+00     9.48683E-01     1.00000E+00
    # 7.07107E-01     3.16228E-01     2.00000E+00
    print(d)
    # [2.0, -4.999999999999999, 1.9999999999999996]
    print()

    print()


def test_householder():
    print('==========Test Householder==========')

    print('----------Case 1----------')
    n = 4
    A = [1, 2, 0, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    b = [3, 11, 10, 6]
    d = [0 for _ in range(n)]

    qr(A, d, n)
    householder(A, d, b, n)
    print(b)
    # [1.0000000000000162, 0.9999999999999915, 1.000000000000015, 1.0000000000000098]
    print()

    print()


if __name__ == '__main__':
    test_lu()
    test_gauss()
    test_qr()
    test_householder()
