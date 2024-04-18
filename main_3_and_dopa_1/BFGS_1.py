import time
import numpy as np
import scipy.optimize as sp


class Function:
    def __init__(self, fn, grad, string_format):
        self.fn = fn
        self.grad = grad
        self.string_format = string_format


class Result:
    def __init__(self, start_point, x, value, delta_time, nit, method_name, func):
        self.start_point = start_point
        self.x = x
        self.value = value
        self.delta_time = delta_time
        self.nit = nit
        self.method_name = method_name
        self.func = func


def bfgs(function, x0, iterations=10_000, eps=10 ** -9):
    start_time = time.time()
    f = function.fn
    g = function.grad
    xk = x0
    I = np.identity(xk.size)
    Hk = I
    cur_iter = 0

    while True:
        # compute search direction
        gk = g(xk)
        pk = -Hk.dot(gk)

        # obtain step length by line search
        alpha = sp.line_search(f, g, xk, pk)[0]
        if alpha is None:
            alpha = 1

        # update x
        xk1 = xk + alpha * pk
        gk1 = g(xk1)

        # define sk and yk for convenience
        sk = xk1 - xk
        yk = gk1 - gk

        # compute next H_{k+1} by BFGS update
        rho_k = float(1.0 / yk.dot(sk))

        Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        if cur_iter == iterations or np.linalg.norm(xk1 - xk) < eps:
            cur_iter += 1
            xk = xk1
            break

        Hk = Hk1
        xk = xk1
        cur_iter += 1

    end_time = time.time()
    return Result(x0, xk, f(xk), end_time - start_time, cur_iter, "BFGS", function.string_format)


def create_log(data, file_name):
    with open(file_name, 'a') as file:
        file.write(f"Calculation method: {data.method_name}\n")
        file.write(f"Computed function: {data.func}\n")
        file.write(f"Start point: {list(data.start_point)}\n")
        file.write(f"Iterations: {data.nit}\n")
        file.write(f"Result args: {list(data.x)}\n")
        file.write(f"Result value: {data.value}\n")
        file.write(f"Execution time (seconds): {data.delta_time}\n\n")


def clear_file(file_name):
    with open(file_name, 'w'):
        pass


def test(function, start_points, name_log):
    clear_file(name_log)
    for x0 in start_points:
        result = bfgs(function, x0)
        create_log(result, name_log)


if __name__ == '__main__':
    start_points = (np.array([0, 0]),
                    np.array([10, 10]),
                    np.array([50, 100]))

    rosenFunction = Function(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
                             lambda x: np.array(
                                 [-2 + 2 * x[0] - 400 * x[0] * x[1] + 400 * x[0] ** 3, 200 * (x[1] - x[0] ** 2)]),
                             "100 * (y - x ^ 2) ^ 2 + (1 - x ^ 2)")

    expFunction = Function(lambda x: np.sin(x[0]) - np.cos(x[1]) + x[0] ** 2,
                           lambda x: np.array([np.cos(x[0]) + 2 * x[0], np.sin(x[1])]),
                           "sin(x) - cos(y) + x ^ 2")

    test(rosenFunction, start_points, "./logs/rosen.log")
    test(expFunction, start_points, "./logs/exp.log")

