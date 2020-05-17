from utils import *
from math import sqrt


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

        if max_t == 0:
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
    # TODO(Rewrite)
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
    output(A, n)
    # 3.000000        7.000000        1.000000        0.000000
    # 0.333333        2.666667        4.666667        -5.000000
    # 0.333333        -0.125000       4.000000        -0.625000
    # 0.666667        0.500000        -0.750000       4.031250
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
    A = [12, -51, 4, 6, 167, -68, -4, 24, -41]
    d = [0 for _ in range(n)]

    qr(A, d, n)
    output(A, n)
    print(d)


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
    # test_lu()
    # test_gauss()
    test_qr()
    # test_householder()
