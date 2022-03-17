import copy


def foo(X):
    X[0] = 12

if __name__ == "__main__":
    A = [11, 12, 13, 14]
    print(A)
    foo(copy.deepcopy(A))
    print(A)