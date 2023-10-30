import time

from geneticalgorithm import geneticalgorithm as ga
import numpy as np

def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

varbound = np.array([[-5, 5]]*2) # Определение диапазона переменных

model = ga(function=rosenbrock, dimension=2, variable_type='real', variable_boundaries=varbound) # Создание модели генетического алгоритма

start = time.time()
model.run()
elapsed_time = time.time() - start

# Вывод результатов
print("=========================================")
print("Математическое ожидание значения функции соответствия: ", model.best_function)
print("Математическое ожидание решения: ", model.best_variable)
print("Дисперсия значения функции соответствия: ", np.var(model.report))
print("Затраченное время: {} сек".format(elapsed_time))
print("=========================================")