"""
@project: Numerical-and-Symbolic-Calculations
@author: jiahao Shen
@file: utils.py
@ide: Visual Studio Code
"""
from math import sqrt, inf


def output(a, m, n):
    """输出矩阵
    @param a: 按行优先次序存放
    @param m: 矩阵行数
    @param n: 矩阵列数
    @return:
    """
    for i in range(m):
        for j in range(n):
            print('%.5E' % a[i * n + j], end='\t')
        print()


def norm(a, ord=2):
    """计算范数
    @param a: 向量
    @param ord: 1范数, 2范数, 正无穷范数, 负无穷范数
    @return: Float
    """
    if ord == 1:
        return sum(abs(x) for x in a)
    elif ord == 2:
        return sqrt(sum(abs(x) ** 2 for x in a))
    elif ord == inf:
        return max(abs(x) for x in a)
    elif ord == -inf:
        return min(abs(x) for x in a)


def identity(n):
    """生成单位矩阵
    @param n: 矩阵维数
    @return: Matrix
    """
    one = [0 for _ in range(n * n)]
    for i in range(n):
        one[i * n + i] = 1

    return one


def outer(a, b):
    """向量外积
    @param a: 列向量
    @param b: 行向量
    @return: Matrix
    """
    m, n = len(a), len(b)
    res = [0 for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            res[i * n + j] = a[i] * b[j]
    return res


def multiply(a, b, m, p, n):
    """矩阵乘法
    @param a: 矩阵A
    @param b: 矩阵B
    @param m: 矩阵A的行
    @param p: 矩阵A的列, 矩阵B的行
    @param n: 矩阵B的列
    @return: Matrix
    """
    res = [0 for _ in range(m * n)]

    for i in range(m):
        for j in range(p):
            if abs(a[i * p + j]) > 1e-9:
                for k in range(n):
                    if abs(b[j * n + k]) > 1e-9:
                        res[i * n + k] += (a[i * p + j] * b[j * n + k])

    return res


def test_norm():
    print('==========Test Norm==========')

    print('----------Case 1----------')
    a = [1, 2, -3]
    print(norm(a, 1))
    # 6
    print(norm(a, 2))
    # 3.7416573867739413
    print(norm(a, inf))
    # 3
    print(norm(a, -inf))
    # 1
    print()

    print('----------Case 2----------')
    a = [1+2j, 2+3j, -3+5j]
    print(norm(a))
    # 7.211102550927978
    print()

    print()


def test_outer():
    print('==========Test Outer==========')

    print('----------Case 1----------')
    m = 3
    a = [1, 2, 3]
    output(outer(a, a), m, m)
    print()

    print('----------Case 2----------')
    m = 3
    a = [2, 3, 4]
    n = 2
    b = [1, 6]
    output(outer(a, b), m, n)
    print()

    print()


def test_multiply():
    print('==========Test Multiply==========')

    print('----------Case 1----------')
    a = [5, 2, 4,
         3, 8, 2,
         6, 0, 4,
         0, 1, 6]
    b = [2, 4,
         1, 3,
         3, 2]

    output(multiply(a, b, 4, 3, 2), 4, 2)
    print()
    print()


if __name__ == '__main__':
    test_norm()
    test_outer()
    test_multiply()
