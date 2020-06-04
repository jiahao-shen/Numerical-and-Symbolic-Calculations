"""
@project: Numerical-and-Symbolic-Calculations
@author: jiahao Shen
@file: utils.py
@ide: Visual Studio Code
"""
from math import sqrt, inf


def output(a):
    """打印矩阵
    @param a: 按行优先次序存放
    @return:
    """
    m, n = len(a), len(a[0])
    for i in range(m):
        for j in range(n):
            print('%.5E' % a[i][j], end='\t')
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
    one = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        one[i][i] = 1

    return one


def outer(a, b):
    """向量外积
    @param a: 列向量
    @param b: 行向量
    @return: Matrix
    """
    m, n = len(a), len(b)
    res = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            res[i][j] = a[i] * b[j]
    return res


def multiply(a, b):
    """矩阵乘法
    @param a: 矩阵A
    @param b: 矩阵B
    @return: Matrix
    """
    m, p, n = len(a), len(a[0]), len(b[0])
    res = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for k in range(p):
            if abs(a[i][k]) > 1e-9:
                for j in range(n):
                    if abs(b[k][j]) > 1e-9:
                        res[i][j] += (a[i][k] * b[k][j])

    return res


def dot(a, b):
    """矩阵向量乘法
    @param a: 矩阵A
    @param b: 向量b
    @return: Vector
    """
    m, n = len(a), len(a[0])
    res = [0 for _ in range(m)]

    for i in range(m):
        for j in range(n):
            res[i] += (a[i][j] * b[j])

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
    output(outer(a, a))
    # 1.00000E+00     2.00000E+00     3.00000E+00
    # 2.00000E+00     4.00000E+00     6.00000E+00
    # 3.00000E+00     6.00000E+00     9.00000E+00
    print()

    print('----------Case 2----------')
    m = 3
    a = [2, 3, 4]
    n = 2
    b = [1, 6]
    output(outer(a, b))
    # 2.00000E+00     1.20000E+01
    # 3.00000E+00     1.80000E+01
    # 4.00000E+00     2.40000E+01
    print()

    print()


def test_multiply():
    print('==========Test Multiply==========')

    print('----------Case 1----------')
    a = [[5, 2, 4],
         [3, 8, 2],
         [6, 0, 4],
         [0, 1, 6]]
    b = [[2, 4],
         [1, 3],
         [3, 2]]

    output(multiply(a, b))
    # 2.40000E+01     3.40000E+01
    # 2.00000E+01     4.00000E+01
    # 2.40000E+01     3.20000E+01
    # 1.90000E+01     1.50000E+01
    print()
    print()


def test_dot():
    print('==========Test Dot==========')

    print('----------Case 1----------')
    a = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    b = [1, 1, 1]

    print(dot(a, b))
    # [3, 3, 3]
    print()
    print()


if __name__ == '__main__':
    test_norm()
    test_outer()
    test_multiply()
    test_dot()
