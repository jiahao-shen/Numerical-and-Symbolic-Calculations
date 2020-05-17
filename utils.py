from math import sqrt


def output(a, m, n):
    for i in range(m):
        for j in range(n):
            print('{:.6f}'.format(a[i * n + j]), end='\t')
        print()


def norm(a, ord=2):
    if ord == 0:
        return max(abs(x) for x in a)
    if ord == 1:
        return sum(abs(x) for x in a)
    if ord == 2:
        return sqrt(sum(x ** 2 for x in a))


def transpose(a, n):
    for i in range(n):
        for j in range(i + 1, n):
            a[i * n + j], a[j * n + i] = a[j * n + i], a[i * n + j]


def identity(n):
    one = [0 for _ in range(n * n)]
    for i in range(n):
        one[i * n + i] = 1

    return one


def outer(a, b, n):
    res = [0 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            res[i * n + j] = a[j] * b[i]
    return res


def dot(a, b, m, p, n):
    res = [0 for _ in range(m * n)]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                res[i * n + j] += a[i * p + k] * b[k * n + j]

    return res


def test_norm():
    print('==========Test Norm==========')

    print('----------Case 1----------')
    a = [1, 2, -3]
    print(norm(a, 0))
    # 3
    print(norm(a, 1))
    # 6
    print(norm(a, 2))
    # 3.7416573867739413
    print()

    print()


def test_outer():
    print('==========Test Outer==========')

    print('----------Case 1----------')
    n = 3
    a = [1, 2, 3]
    output(outer(a, a, n), n)
    print()

    print()


def test_dot():
    print('==========Test Dot==========')

    print('----------Case 1----------')
    a = [5, 2, 4,
         3, 8, 2,
         6, 0, 4,
         0, 1, 6]
    b = [2, 4,
         1, 3,
         3, 2]

    output(dot(a, b, 4, 3, 2), 4, 2)
    print()
    print()


if __name__ == '__main__':
    # test_norm()
    # test_outer()
    test_dot()
