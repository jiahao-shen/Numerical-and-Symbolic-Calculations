"""
@project: Numerical-and-Symbolic-Calculations
@auther: jiahao Shen
@file: linear_equations.py
@ide: Visual Studio Code
@time: 2020-05-08
"""
from math import fabs, sqrt


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
        max_t = fabs(a[i * n + i])
        t = i

        for j in range(i + 1, n):
            if fabs(a[j * n + i]) > max_t:
                max_t = fabs(a[j * n + i])
                t = j

        if max_t == 0:
            return True

        for j in range(n):
            a[i * n + j], a[t * n + j] = a[t * n + j], a[i * n + j]

        pivot[i] = t

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
    for i in range(n - 1):
        b[i], b[p[i]] = b[p[i]], b[i]

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

        for k in range(i + 1, n):
            for j in range(i, n):
                t = 0
                for l in range(i, n):
                    t += (a[j * n + i] * a[l * n + i] * a[l * n + k])
                temp[j] = a[j * n + k] - 2 * t

            for j in range(i, n):
                a[j * n + k] = temp[j]

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


def hilbert(n):
    """生成希尔伯特矩阵方程组Hx=b
    @param n: 矩阵维数
    @return: h, b
    """
    h = [0 for _ in range(n * n)]
    b = [0 for _ in range(n)]

    for i in range(n):
        for j in range(n):
            h[i * n + j] = 1 / (i + j + 1)
            b[i] += h[i * n + j]

    return h, b


def ouput(a, n):
    for i in range(n):
        for j in range(n):
            print('{:.6f}'.format(a[i * n + j]), end='\t')
        print()


def test_lu():
    n = 4
    a = [1, 2, 15/4, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    pivot = list(range(n))

    lu(a, pivot, n)

    ouput(a, n)
    print(pivot)

    print('-------------------')

    n = 6
    a, b = hilbert(n)
    pivot = list(range(n))

    lu(a, pivot, n)

    ouput(a, n)
    print(pivot)


def test_gauss():
    n = 4
    a = [1, 2, 0, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    b = [3, 11, 10, 6]
    pivot = list(range(n))

    lu(a, pivot, n)
    gauss(a, pivot, b, n)
    print(b)

    n = 8
    a, b = hilbert(n)
    pivot = list(range(n))

    lu(a, pivot, n)
    gauss(a, pivot, b, n)
    print(b)


def test_qr():
    n = 6
    a, b = hilbert(n)
    d = [0 for _ in range(n)]

    qr(a, d, n)
    ouput(a, n)
    print(d)


def test_household():
    n = 6
    a, b = hilbert(n)
    d = [0 for _ in range(n)]

    qr(a, d, n)
    householder(a, d, b, n)
    print(b)

    n = 4
    a = [1, 2, 0, 0,
         3, 7, 1, 0,
         2, 6, 0, 2,
         1, 5, 5, -5]
    b = [3, 11, 10, 6]
    d = [0 for _ in range(n)]

    qr(a, d, n)
    householder(a, d, b, n)
    print(b)


if __name__ == '__main__':
    test_lu()
    test_gauss()
    test_qr()
    test_household()
