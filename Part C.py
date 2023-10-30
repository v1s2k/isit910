import time

import numpy as np
from sklearn import datasets, svm, neighbors, ensemble
from sklearn.model_selection import train_test_split
from geneticalgorithm import geneticalgorithm as ga

# Загрузка данных
data = datasets.load_iris()
X = data.data
y = data.target

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Функция для оптимизации (оценка точности модели)
def fitness_function(params):
    # Разделение параметров
    svm_C, knn_n_neighbors, rf_n_estimators = params

    # Создание моделей
    svm_model = svm.SVC(C=svm_C)
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=int(knn_n_neighbors))
    rf_model = ensemble.RandomForestClassifier(n_estimators=int(rf_n_estimators))

    # Обучение моделей
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Оценка точности моделей на тестовой выборке
    svm_accuracy = svm_model.score(X_test, y_test)
    knn_accuracy = knn_model.score(X_test, y_test)
    rf_accuracy = rf_model.score(X_test, y_test)

    # Вернуть среднюю точность
    return np.mean([svm_accuracy, knn_accuracy, rf_accuracy])


# Задание диапазона переменных
varbound = np.array([[1,10]]*3)

# Создание экземпляра генетического алгоритма
model = ga(function=fitness_function, dimension=3, variable_type='int', variable_boundaries=varbound)


start = time.time()
model.run()
elapsed_time = time.time() - start

# Вывод результатов
print("Оптимальные значения параметров:")
print("Затраченное время: {} сек".format(elapsed_time))
print("SVM C:", model.best_variable[0])
print("KNN n_neighbors:", model.best_variable[1])
print("RF n_estimators:", model.best_variable[2])
print("Лучшая точность:", model.best_function)