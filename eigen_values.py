from utils import *
from random import random
from linear_equations import *
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
        env_new = dot(a, env, n, n, 1)

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
        A = a[:]
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
        t, t_i, t_j = 0, -1, -1

        for i in range(n):
            for j in range(n):
                if i != j and abs(a[i * n + j]) > t:
                    t = abs(a[i * n + j])
                    t_i, t_j = i, j

        if abs(t) < 1e-9:
            return True

        if a[t_i * n + t_i] == a[t_j * n + t_j]:
            theta = pi / 4
        else:
            theta = atan(2 * a[t_i * n + t_j] /
                         (a[t_i * n + t_i] - a[t_j * n + t_j])) / 2

        a_new = a[:]

        a_new[t_i * n + t_i] = a[t_i * n + t_i] * (cos(theta) ** 2) + a[t_j * n + t_j] * (
            sin(theta) ** 2) + 2 * a[t_i * n + t_j] * cos(theta) * sin(theta)
        a_new[t_j * n + t_j] = a[t_i * n + t_i] * (sin(theta) ** 2) + a[t_j * n + t_j] * (
            cos(theta) ** 2) - 2 * a[t_i * n + t_j] * cos(theta) * sin(theta)
        a_new[t_i * n + t_j] = (a[t_j * n + t_j] - a[t_i * n + t_i]) * \
            sin(2 * theta) / 2 + a[t_i * n + t_j] * cos(2 * theta)
        a_new[t_j * n + t_i] = a_new[t_i * n + t_j]

        for i in range(n):
            if i != t_i and i != t_j:
                a_row = a_new[t_i * n + i]
                a_col = a_new[t_j * n + i]
                a_new[t_i * n + i] = a_row * cos(theta) + a_col * sin(theta)
                a_new[i * n + t_i] = a_new[t_i * n + i]
                a_new[t_j * n + i] = a_col * cos(theta) - a_row * sin(theta)
                a_new[i * n + t_j] = a_new[t_j * n + i]

        for i in range(n):
            for j in range(n):
                a[i * n + j] = a_new[i * n + j]

        for i in range(n):
            env[i] = a[i * n + i]

    return True


def gauss_hessen(a, n):
    """用Gauss相似变换将矩阵转换为Hessenberg矩阵
    @param a: 按行优先次序存放
    @param n: 矩阵维数
    """
    for k in range(1, n - 1):
        i = k
        for j in range(k + 1, n):
            if abs(a[j * n + k - 1]) > abs(a[k * n + k - 1]):
                i = j

        temp = a[i * n + k - 1]

        if abs(temp) < 1e-9:
            return True

        if i != k:
            for j in range(k - 1, n):
                a[i * n + j], a[k * n + j] = a[k * n + j], a[i * n + j]
            for j in range(n):
                a[j * n + i], a[j * n + k] = a[j * n + k], a[j * n + i]
        for i in range(k + 1, n):
            m = a[i * n + k - 1] / temp
            a[i * n + k - 1] = 0
            for j in range(k, n):
                a[i * n + j] -= m * a[k * n + j]
            for j in range(n):
                a[j * n + k] += m * a[j * n + i]

    return False


class Eigen(object):

    def __init__(self):
        pass

    def qr_aux(self, i, j):
        """
        """
        if j - i <= 2:
            for k in range(i, j):
                self.en[k] = self.a[k * self.n + k]
            return

        A = [0 for _ in range((j - i) * (j - i))]
        for k in range(j - i):
            for l in range(j - i):
                A[k * (j - i) + l] = self.a[(i + k) * self.n + (i + l)]

        for _ in range(100):
            Q, R = self.qr_solve(A, j - i)
            A = dot(R, Q, j - i, j - i, j - i)

            for k in range(i + 1, j):
                if abs(A[k * (j - i) + k - 1]) < 1e-9:
                    for k in range(j - i):
                        for l in range(j - i):
                            self.a[(i + k) * self.n + (i + l)
                                   ] = A[k * (j - i) + l]

                    self.qr_aux(i, k)
                    self.qr_aux(k, j)
                    return False

        for k in range(i, j):
            self.en[k] = A[(k - i) * (j - i) + (k - i)]

        return False

    def qr_eng(self, result, h, m):
        self.a = h
        self.n = m
        self.en = result

        self.qr_aux(0, self.n)

    def qr_solve(self, A, n):
        d = [0 for _ in range(n)]

        qr(A, d, n)

        R = [0 for _ in range(n * n)]
        for i in range(n):
            R[i * n + i] = d[i]
            for j in range(i + 1, n):
                R[i * n + j] = A[i * n + j]

        Q = identity(n)
        for i in range(n - 1):
            h = identity(n)
            v = []
            for j in range(i, n):
                v.append(A[j * n + i])
            v2 = outer(v, v, len(v))

            for j in range(len(v)):
                for k in range(len(v)):
                    h[(j + i) * n + k + i] -= 2 * v2[j * len(v) + k]

            Q = dot(Q, h, n, n, n)

        return Q, R


def test_power_eng():
    print('==========Test Power Eng==========')
    print('----------Case 1----------')
    n = 3
    a = [-4, 14, 0,
         -5, 13, 0,
         -1, 0, 2]
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
    a = [2, 1, 0, 0,
         1, 2, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 2]
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
    a = [2, -1, 0,
         0, 2, -1,
         0, -1, 2]
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
    a = [2, -1, 0,
         -1, 2, -1,
         0, -1, 2]
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
    a = [-4, 14, 0,
         -5, 13, 0,
         -1, 0, 2]
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
    a = [-1, 1, 0,
         -4, 3, 0,
         1, 0, 2]
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
    a = [4, 2, 2,
         2, 5, 1,
         2, 1, 6]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [2.12592446854474, 8.387619058475414, 4.486456472979847]
    print()

    print('----------Case 2----------')
    n = 4
    a = [1, 1, 1, 1,
         1, 2, 3, 4,
         1, 3, 6, 10,
         1, 4, 10, 20]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.4538345500256738, 2.2034461676473147, 0.038016015229141276, 26.304703267097885]

    print('----------Case 3----------')
    n = 4
    a = [4, -30, 60, -35,
         -30, 300, -675, 420,
         60, -675, 1620, -1050,
         -35, 420, -1050, 700]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.16664286117190066, 37.10149136512766, 2585.253810928919, 1.4780548447781765]
    print()

    print('----------Case 4----------')
    n = 3
    a = [3.5, -6, 5,
         -6, 8.5, -9,
         5, -9, 8.5]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [0.4659302062458505, -0.9340137468008786, 20.968083540555032]
    print()

    print('----------Case 5----------')
    n = 3
    a = [6, 2, 4,
         2, 3, 2,
         4, 2, 6]
    env = [0 for _ in range(n)]

    jacobi_eng(env, a, n)
    print(env)
    # [11.0, 2.0, 1.9999999999999998]
    print()


def test_gauss_hessen():
    print('==========Test Gauss Hessenberg==========')
    print('----------Case 1----------')
    n = 4
    A = [9, 18, 9, -27,
         18, 45, 0, -45,
         9, 0, 126, 9,
         -27, -45, 9, 135]

    gauss_hessen(A, n)
    output(A, n, n)
    print()

    print('----------Case 2----------')
    n = 4
    A = [54, 40, 10, 76,
         47, 20, 94, 49,
         26, 80, 94, 70,
         3, 92, 83, 45]

    gauss_hessen(A, n)
    output(A, n, n)
    print()

    print('----------Case 3----------')
    n = 3
    A = [-149, -50, -154,
         537, 180, 546,
         -27, -9, -25]

    gauss_hessen(A, n)
    output(A, n, n)
    print()

    print('----------Case 4----------')
    n = 3
    A = [1, 3, 4,
         3, 2, 1,
         4, 1, 3]

    gauss_hessen(A, n)
    output(A, n, n)
    print()

    print('----------Case 5----------')
    n = 4
    A = [8, 6, 10, 10,
         9, 1, 10, 5,
         1, 3, 1, 8,
         10, 6, 10, 1]

    gauss_hessen(A, n)
    output(A, n, n)
    print()


def test_hessen_qr():
    print('==========Test QR Iteration==========')
    print('----------Case 1----------')
    n = 3
    A = [5, -3, 2,
         6, -4, 4,
         4, -4, 5]

    gauss_hessen(A, n)
    output(A, n, n)

    env = [0 for _ in range(n)]
    eig = Eigen()
    eig.qr_eng(env, A, n)
    print(env)
    # [3.0000000000000018, 1.9999999999999991, 1.0000000000000009]
    print()

    print('----------Case 2----------')
    n = 5
    A = [5, -5/3, 2, 4, 5,
         6, -4/3, 4, 5, 6,
         0, 2/9, 7/3, 6, 7,
         0, 0, 0, 7, 8,
         0, 0, 0, 0, 5]

    env = [0 for _ in range(n)]
    eig = Eigen()
    eig.qr_eng(env, A, n)
    # [3.0000000000000018, 1.9999999999999991, 1.0000000000000016, 7.0, 5.0]
    print(env)
    print()


if __name__ == '__main__':
    test_power_eng()
    test_inv_power_eng()
    test_jacobi_eng()
    test_gauss_hessen()
    test_hessen_qr()
