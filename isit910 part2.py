import time

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from matplotlib import pyplot as plt


def rosenbrock(X):
    return sum([100 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2
                for i in range(len(X) - 1)])

# Определение значения дисперсии
def objective_function(x):
    func_value = rosenbrock(x)
    return np.var(func_value)

varbound = np.array([[1,10]]*3)  # Задаем границы переменных массив из трех элементов, каждый из которых является массивом [1, 10].
# Это означает, что у нас есть три переменные, и каждая из них имеет границы от 1 до 10

# Генетический алгоритм
algorithm_param = {'max_num_iteration': 100,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'crossover_probability': 0.6,
                        'parents_portion': 0.5

                   }

model = ga(function=objective_function, dimension=3, variable_type='real', algorithm_parameters=algorithm_param, variable_boundaries=varbound)#dimension - это число, указывающее количество переменных, которые участвуют в генетическом алгоритме.
# В этом случае, dimension=3, что означает, что у нас три переменные.
start = time.time()
model.run()
elapsed_time = time.time() - start




# Вывод результатов
print("=========================================")
print("Массив переменных для генетического алгоритма.: ",varbound)
print("Минимальное значение дисперсии:", model.best_function)
print("Оптимальные значения параметров:", model.best_variable)
print("Затраченное время: {} сек".format(elapsed_time))
print("=========================================")


# Визуализация функции Розенброка
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('Функция Розенброка')
plt.show()

