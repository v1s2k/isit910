import math
import numpy as np
from scipy.optimize import minimize, approx_fprime, optimize
from scipy.optimize import minimize, optimize

# Определение функции Розенброка
def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)


def rosen_der (x):
    xm = x [1: -1]
    xm_m1 = x [: - 2]
    xm_p1 = x [2:]
    der = np.zeros_like (x)
    der [1: -1] = 200 * (xm-xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1-xm)
    der [0] = -400 * x [0] * (x [1] -x [0] ** 2) - 2 * (1-x [0])
    der [-1] = 200 * (x [-1] -x [-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H



# Выполнение оптимизации с использованием классического алгоритма (Ньютона)
newton_results = []
newton_resultsr = []
for _ in range(100):
    x0 = np.array([1.3, 2])
    x0r = np.random.randn(5)
    resultr = minimize(rosen, x0r, method='Newton-CG',
                   jac=rosen_der, hess=rosen_hess)
    valuer = resultr.fun  # Значение функции соответствия
    newton_resultsr.append(valuer)
    result = minimize(rosen, x0, method='Newton-CG',
                      jac=rosen_der, hess=rosen_hess)
    value = result.fun  # Значение функции соответствия
    newton_results.append(value)
newton_meanr = np.mean(newton_resultsr)
newton_variancer = np.var(newton_resultsr)
newton_mean = np.mean(newton_results)
newton_variance = np.var(newton_results)

# Выполнение оптимизации с использованием эволюционного алгоритма
print("Алгоритм сопряженных градиентов (Ньютона) для случайных:")
print("Математическое ожидание:", newton_meanr)
print("Дисперсия:", newton_variancer)
print(x0r)
print("Алгоритм сопряженных градиентов (Ньютона) для 1.3, 2 :")
print("Математическое ожидание:", newton_mean)
print("Дисперсия:", newton_variance)
print(x0)

