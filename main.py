import time

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# Определяем функцию Розенброка
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Определяем функцию для вычисления значения функции соответствия
def fitness_func(x):
    return -rosenbrock(x)

# Функция для выполнения одного прогона генетического алгоритма
def run_genetic_algorithm():
    algorithm_param = {'max_num_iteration': None,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'mutation_type': 'uniform_by_center',
                       'selection_type': 'roulette',
                       'max_iteration_without_improv': 100}


    varbound = np.array([[0, 10]] * 3)
    model = ga(function=fitness_func, dimension=3, variable_type='real', algorithm_parameters=algorithm_param,variable_boundaries=varbound)
    model.run()

    return model.best_function

# Выполняем 100 прогонов генетического алгоритма и сохраняем значения функций соответствия
fitness_values = []
for _ in range(10):
    fitness_values.append(run_genetic_algorithm())

# Вычисляем математическое ожидание и дисперсию для финального значения функции соответствия
mean_fitness = np.mean(fitness_values)
var_fitness = np.var(fitness_values)

print("Математическое ожидание для финального значения функции соответствия:", mean_fitness)
print("Дисперсия для финального значения функции соответствия:", var_fitness)

# Оцениваем время для достижения априори известного значения глобального экстремума
start_point = np.ones(5) * -2
target_fitness = fitness_func(start_point)

start_time = time.time()

while True:
    current_fitness = run_genetic_algorithm()
    if current_fitness <= target_fitness:
        break

total_time = time.time() - start_time

print("Время для достижения априори известного значения глобального экстремума:", total_time)

# Визуализация оптимизируемой функции
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(np.array([X, Y]))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Оптимизируемая функция Розенброка')
plt.show()

# Визуализация значений функции соответствия в зависимости от числа поколений
generations = np.arange(1, 11)
fitness_values = []

for generation in generations:
    fitness_values.append(run_genetic_algorithm())

plt.plot(generations, fitness_values)
plt.xlabel('Число поколений')
plt.ylabel('Значение функции')
plt.show()