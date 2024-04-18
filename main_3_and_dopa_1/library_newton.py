import time
import numpy as np
from scipy.optimize import minimize

from BFGS_1 import Result, Function, clear_file, create_log


def testLibrary(function, start_points, name_log, invokeFunction):
    clear_file(name_log)
    for x0 in start_points:
        result = invokeFunction(function, x0)
        create_log(result, name_log)


def testLibraryNewton(function, start_points, name_log):
    testLibrary(function, start_points, name_log, library_newton_cg)


def testLibraryBFGS(function, start_points, name_log):
    testLibrary(function, start_points, name_log, library_bfgs)


def library_newton_cg(function, x0):
    start_time = time.time()
    res = minimize(function.fn, x0, method='Newton-CG', jac=function.grad, tol=1e-9)
    end_time = time.time()
    return Result(x0, res.x, function.fn(res.x), end_time - start_time, res.nit, "Newton-CG", function.string_format)


def library_bfgs(function, x0):
    start_time = time.time()
    res = minimize(function.fn, x0, method='BFGS', jac='2-point', tol=1e-9)
    end_time = time.time()
    return Result(x0, res.x, function.fn(res.x), end_time - start_time, res.nit, "Library BFGS", function.string_format)


if __name__ == "__main__":
    start_points = (np.array([0, 0]),
                    np.array([10, 10]),
                    np.array([50, 100]))

    rosenFunction = Function(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
                             lambda x: np.array(
                                 [-2 + 2 * x[0] - 400 * x[0] * x[1] + 400 * x[0] ** 3, 200 * (x[1] - x[0] ** 2)]),
                             "100 * (y - x ^ 2) ^ 2 + (1 - x ^ 2)")

    sinFunction = Function(lambda x: np.sin(x[0]) - np.cos(x[1]) + x[0] ** 2,
                           lambda x: np.array([np.cos(x[0]) + 2 * x[0], np.sin(x[1])]),
                           "sin(x) - cos(y) + x ^ 2")

    testLibraryNewton(rosenFunction, start_points, "./logs/library_newton_rosen.log")
    testLibraryBFGS(rosenFunction, start_points, "./logs/library_BFGS_rosen.log")
    testLibraryNewton(sinFunction, start_points, "./logs/library_newton_sin.log")
    testLibraryBFGS(sinFunction, start_points, "./logs/library_BFGS_sin.log")
