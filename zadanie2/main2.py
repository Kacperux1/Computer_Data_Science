import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import algorithm

# Wczytanie danych
class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.sepal_length = float(sepal_length)
        self.sepal_width = float(sepal_width)
        self.petal_length = float(petal_length)
        self.petal_width = float(petal_width)


init_data = pd.read_csv("data2test.csv")

# Normalizacja danych
scaler = MinMaxScaler()
scaler.fit(init_data)
init_data_nor = scaler.transform(init_data)

# WCSS i liczba iteracji dla różnych liczby klastrów
wcss = []
iterations = []

for i in range(2, 11):
    result = algorithm.CustomKMeans(i, init_data_nor, 0.000001)
    wcss.append(result.wcss)
    iterations.append(result.iteration_number)

# Wykres WCSS
plt.plot(range(2, 11), wcss, color="black")
plt.title('WCSS vs. Liczba klastrów')
plt.xlabel('Liczba klastrów')
plt.ylabel('WCSS')
plt.grid()
plt.show()

print("WCSS:", wcss)
print("Iterations:", iterations)

result = algorithm.CustomKMeans(3, init_data_nor, 0.000001)



init_data["group"] = result.groups

group0 = init_data[init_data.group == 0]
group1 = init_data[init_data.group == 1]
group2 = init_data[init_data.group == 2]


centers = np.array(result.centers)



centers[:, 0] = (
           centers[:, 0] * (max(init_data["sepal_length"]) - min(init_data["sepal_length"])) + min(
        init_data["sepal_length"]))
centers[:, 1] = (
            centers[:, 1] * (max(init_data["sepal_width"]) - min(init_data["sepal_width"])) + min(
        init_data["sepal_width"]))
centers[:, 2] = (
            centers[:, 2] * (max(init_data["petal_length"]) - min(init_data["petal_length"])) + min(
        init_data["petal_length"]))
centers[:, 3] = (
            centers[:, 3] * (max(init_data["petal_width"]) - min(init_data["petal_width"])) + min(
        init_data["petal_width"]))


plt.scatter(group0.sepal_length, group0.sepal_width, marker='o', facecolors='none', edgecolors='orange',)
plt.scatter(group1.sepal_length, group1.sepal_width, marker='o', facecolors='none', edgecolors='red')
plt.scatter(group2.sepal_length, group2.sepal_width, marker='o', facecolors='none', edgecolors='green')
plt.scatter([center[0] for center in centers], [center[1] for center in centers], color='black', marker='*', s=200)
plt.title('Długość działki kielicha a szerokość działki kielicha')
plt.xlabel('Długość działki kielicha')
plt.ylabel('Szerokość działki kielicha')
plt.show()

plt.scatter(group0.sepal_length, group0.sepal_width, color='orange')
plt.scatter(group1.sepal_length, group1.sepal_width, color='red')
plt.scatter(group2.sepal_length, group2.sepal_width, color='green')
plt.scatter(centers[:, 0],centers[:, 1], color='black', marker='*', s=200)
plt.title('dłogosc dzialki kielicha a szerokosc dzialki kielicha')
plt.xlabel('dłogosc dzialki kielicha')
plt.ylabel('szerokosc dzialki kielicha')
plt.show()


plt.scatter(group0.sepal_length, group0.petal_length, color='orange')
plt.scatter(group1.sepal_length, group1.petal_length, color='red')
plt.scatter(group2.sepal_length, group2.petal_length, color='green')
plt.scatter(centers[:,0], centers[:,2], color='black', marker='*', s=200)
plt.show()

plt.scatter(group0.sepal_length, group0.petal_width, color='orange')
plt.scatter(group1.sepal_length, group1.petal_width, color='red')
plt.scatter(group2.sepal_length, group2.petal_width, color='green')
plt.scatter(centers[:,0], centers[:,3], color='black', marker='*', s=200)
plt.show()

plt.scatter(group0.sepal_width, group0.petal_length, color='orange')
plt.scatter(group1.sepal_width, group1.petal_length, color='red')
plt.scatter(group2.sepal_width, group2.petal_length, color='green')
plt.scatter(centers[:,1], centers[:,2], color='black', marker='*', s=200)
plt.show()

plt.scatter(group0.sepal_width, group0.petal_width, color='orange')
plt.scatter(group1.sepal_width, group1.petal_width, color='red')
plt.scatter(group2.sepal_width, group2.petal_width, color='green')
plt.scatter(centers[:,1], centers[:,3], color='black', marker='*', s=200)
plt.show()

plt.scatter(group0.petal_length, group0.petal_width, color='orange')
plt.scatter(group1.petal_length, group1.petal_width, color='red')
plt.scatter(group2.petal_length, group2.petal_width, color='green')
plt.scatter(centers[:,2], centers[:,3], color='black', marker='*', s=200)
plt.show()

