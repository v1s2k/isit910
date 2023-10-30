import math
import time

import numpy as np
from scipy.optimize import minimize, approx_fprime, optimize
from scipy.optimize import minimize, optimize
from geneticalgorithm import geneticalgorithm as ga

def rosenbrock(X):
    return sum([100 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2
                for i in range(len(X) - 1)])

# Определение значения дисперсии
def objective_function(x):
    func_value = rosenbrock(x)
    return np.var(func_value)


def rosen_der(x):
    xm = x[1: -1]
    xm_m1 = x[: - 2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1: -1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


# Выполнение оптимизации с использованием классического алгоритма (Ньютона)
newton_results = []
newton_resultsr = []
for _ in range(100):
    x0 = np.array([1.3, 2])
    x0r = np.random.randn(5)
    varbound = np.array([[1,10]] * 3)

    # Генетический алгоритм
    algorithm_param = {'max_num_iteration': 100,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'crossover_probability': 0.6,

                       }

    model = ga(function=objective_function, dimension=3, variable_type='real',
               variable_boundaries=varbound)  # dimension - это число, указывающее количество переменных, которые участвуют в генетическом алгоритме.
    # В этом случае, dimension=3, что означает, что у нас три переменные.
    start = time.time()
    model.run()
    elapsed_time = time.time() - start
    valuer = model.fun  # Значение функции соответствия
    newton_resultsr.append(valuer)

newton_meanr = np.mean(newton_resultsr)
newton_variancer = np.var(newton_resultsr)


print("Алгоритм сопряженных градиентов (Ньютона) для случайных:")
print("Математическое ожидание:", newton_meanr)
print("Дисперсия:", newton_variancer)
print(x0r)
