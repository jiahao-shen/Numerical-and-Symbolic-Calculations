import numpy as np
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
        env_new = [0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                env_new[i] += a[i * n + j] * env[j]

        l = norm(env_new)

        for i in range(n):
            env_new[i] /= l

        # if norm([env[i] - env_new[i] for i in range(n)]) <= 1e-6:
        #     print('fuck', _)
            # pld[0] = l
            # return False

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
    env_new = [0 for _ in range(n)]
    for i in range(n):
        env_new[i] = env[i]

    for _ in range(1000):
        A = [0 for _ in range(n * n)]
        for i in range(n):
            for j in range(n):
                A[i * n + j] = a[i * n + j]

        pivot = list(range(n))
        lu(A, pivot, n)
        gauss(A, pivot, env_new, n)

        l = norm(env_new)

        for i in range(n):
            env_new[i] /= l

        # if norm([env[i] - env_new[i] for i in range(n)]) <= 1e-6:
        # pld[0] = 1 / l
        # return False

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
        t, x, y = 0, -1, -1

        for i in range(n):
            for j in range(n):
                if i != j and abs(a[i * n + j]) > t:
                    t = abs(a[i * n + j])
                    x, y = i, j

        if t == 0:
            return True

        if a[x * n + x] == a[y * n + y]:
            theta = pi / 4
        else:
            theta = atan(2 * a[x * n + y] / (a[x * n + x] - a[y * n + y])) / 2

        a_new = [0 for _ in range(n * n)]
        for i in range(n):
            for j in range(n):
                a_new[i * n + j] = a[i * n + j]

        a_new[x * n + x] = a[x * n + x] * (cos(theta) ** 2) + a[y * n + y] * (
            sin(theta) ** 2) + 2 * a[x * n + y] * cos(theta) * sin(theta)
        a_new[y * n + y] = a[x * n + x] * (sin(theta) ** 2) + a[y * n + y] * (
            cos(theta) ** 2) - 2 * a[x * n + y] * cos(theta) * sin(theta)
        a_new[x * n + y] = (a[y * n + y] - a[x * n + x]) * \
            sin(2 * theta) / 2 + a[x * n + y] * cos(2 * theta)
        a_new[y * n + x] = a_new[x * n + y]

        for i in range(n):
            if i != x and i != y:
                a_row = a_new[x * n + i]
                a_col = a_new[y * n + i]
                a_new[x * n + i] = a_row * cos(theta) + a_col * sin(theta)
                a_new[i * n + x] = a_new[x * n + i]
                a_new[y * n + i] = a_col * cos(theta) - a_row * sin(theta)
                a_new[i * n + y] = a_new[y * n + i]

        # if norm([a[i] - a_new[i] for i in range(n * n)]) < 1e-6:
        #     for i in range(n):
        #         env[i] = a[i * n + i]
        #     return False

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
    pass


class Complex(object):

    def __init__(self):
        pass

    def qr_aux(self, i, j):
        """
        """
        return False

    def qr_eng(self, result, h, m):
        self.a = h
        self.n = m
        self.en = result

        self.qr_aux(0, self.n)


def norm(a, ord=2):
    if ord == 0:
        return max(abs(x) for x in a)
    if ord == 1:
        return sum(abs(x) for x in a)
    if ord == 2:
        return sqrt(sum(x ** 2 for x in a))


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
    # [1.0000000000013642]
    # [0.0, 1.9073486328090306e-06, 0.999999999998181]
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
    # [10.999999999999998, 1.9999999999999998, 2.0]
    print()


if __name__ == '__main__':
    test_power_eng()
    test_inv_power_eng()
    test_jacobi_eng()
