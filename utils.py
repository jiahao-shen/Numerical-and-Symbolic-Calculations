from math import sqrt


def output(a, n):
    for i in range(n):
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


if __name__ == '__main__':
    test_norm()
