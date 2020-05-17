from utils import *
from time import time
from linear_equations import *


def calculate_error(b, ans, n):
    r = [0 for _ in range(n)]
    for i in range(n):
        r[i] = b[i] - ans[i]

    print('绝对误差:', norm(r, ord=1))
    print('相对误差:', norm(r, ord=1) / sum(ans) * 100, '%')


def calculate_remnant(A, b, x, n):
    r = [0 for _ in range(n)]
    Ax = dot(A, x, n, n, 1)
    for i in range(n):
        r[i] = Ax[i] - b[i]

    print('残量:', norm(r))
    print('相对残量:', norm(r) / norm(x))


def equation_1(n):
    h = [0 for _ in range(n * n)]
    b = [0 for _ in range(n)]

    for i in range(n):
        for j in range(n):
            h[i * n + j] = 1 / (i + j + 1)
            b[i] += h[i * n + j]

    return h, b


def equation_2(n):
    g = [0 for _ in range(n * n)]
    b = [0 for _ in range(n)]

    for i in range(n):
        g[i * n + n - 1] = 1
        g[i * n + i] = 1

        for j in range(i):
            g[i * n + j] = -1

    for i in range(n):
        for j in range(n):
            b[i] += g[i * n + j]

    return g, b


def equation_3(n):
    g = [0 for _ in range(n * n)]

    for i in range(n):
        g[i * n + n - 1] = 1
        g[i * n + i] = 1

        for j in range(i):
            g[i * n + j] = -1

    b = list(range(2, n + 1)) + [n]

    return g, b


def main_1(n):
    print('n:', n)

    print('-------------Gauss-------------')

    h, b = equation_1(n)
    pivot = list(range(n))

    t = time()
    lu(h, pivot, n)
    gauss(h, pivot, b, n)
    t = time() - t

    print(pivot)
    print('x:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()

    print('-------------Householder-------------')

    h, b = equation_1(n)
    d = [0 for _ in range(n)]

    t = time()
    qr(h, d, n)
    householder(h, d, b, n)
    t = time() - t

    print('x:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()
    print()


def main_2(n):
    print('n:', n)

    print('-------------Gauss-------------')

    h, b = equation_2(n)
    pivot = list(range(n))

    t = time()
    lu(h, pivot, n)
    gauss(h, pivot, b, n)
    t = time() - t

    print('x:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()

    print('-------------Householder-------------')

    h, b = equation_2(n)
    d = [0 for _ in range(n)]

    t = time()
    qr(h, d, n)
    householder(h, d, b, n)
    t = time() - t

    print('x:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()
    print()


def main_3(n):
    print('n:', n)

    print('-------------Gauss-------------')

    h, b = equation_3(n)
    pivot = list(range(n))

    t = time()
    lu(h, pivot, n)
    gauss(h, pivot, b, n)
    t = time() - t

    print('x:', b)
    print('用时:', t * 1000, 'ms')
    H, B = equation_3(n)
    calculate_remnant(H, B, b, n)
    print()

    print('-------------Householder-------------')

    h, b = equation_3(n)
    d = [0 for _ in range(n)]

    t = time()
    qr(h, d, n)
    householder(h, d, b, n)
    t = time() - t

    print('x:', b)
    print('用时:', t * 1000, 'ms')
    H, B = equation_3(n)
    calculate_remnant(H, B, b, n)
    print()
    print()


if __name__ == '__main__':
    print('==================实验1==================')
    main_1(5)
    main_1(10)
    main_1(15)

    print('==================实验2==================')
    main_2(10)
    main_2(30)
    main_2(60)

    print('==================实验3==================')
    main_3(10)
    main_3(30)
    main_3(60)
