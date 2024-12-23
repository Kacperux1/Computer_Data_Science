import pandas as pd
import algorithm
import matplotlib.pyplot as plt

train_data = pd.read_csv("data3_train.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
test_data = pd.read_csv("data3_test.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

columns_to_convert = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
train_data[columns_to_convert] = train_data[columns_to_convert].astype(float)
test_data[columns_to_convert] = test_data[columns_to_convert].astype(float)

train_data["species"] = train_data["species"].astype(int)
test_data["species"] = test_data["species"].astype(int)

outcomes = []
percentages = []
best_result = 0
best_outcome = {}
for i in range(1, 16):
    outcomes.append(algorithm.custom_knn(i, train_data, test_data))
    percentage = float(outcomes[i - 1]["percentage"])

    if percentage > best_result:
        best_result = percentage
        best_outcome = outcomes[i - 1]

for obj in outcomes:
    percentages.append(obj["percentage"]*100)

#wykres procentowy w zależnosci od k dla wszytskich 4 cech naraz
plt.bar(range(1,16), percentages, color='blue', label="Procent")
plt.ylim(80, 102)
plt.xticks(range(1, 16))
plt.title("Sumaryczny wynik klasyfikacji w zależności od k")
plt.xlabel("k (liczba sąsiadów)")
plt.ylabel("Procent [%]")
plt.show()
