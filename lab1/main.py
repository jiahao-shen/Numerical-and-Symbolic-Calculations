from time import time
from copy import deepcopy
from linear_equations import *


def calculate_error(b, ans, n):
    s = 0
    for i in range(n):
        s += fabs(b[i] - ans[i])
    
    print('绝对误差:', s)
    print('相对误差:', s / sum(ans) * 100, '%')

def main_1(n):
    print('n:', n)

    h, b = hilbert(n)
    pivot = [0 for _ in range(n)]

    t = time()
    lu(h, pivot, n)
    gauss(h, pivot, b, n)
    t = time() - t

    print('-------------Gauss-------------')
    print('b:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()

    h, b = hilbert(n)
    d = [0 for _ in range(n)]

    t = time()
    qr(h, d, n)
    householder(h, d, b, n)
    t = time() - t

    print('-------------Householder-------------')
    print('b:', b)
    print('用时:', t * 1000, 'ms')
    ans = [1 for _ in range(n)]
    calculate_error(b, ans, n)
    print()
    print('===============================')


def main_2():
    pass


def main_3():
    pass


if __name__ == '__main__':
    main_1(5)
    main_1(10)
    main_1(15)
