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

result = algorithm.repeat(train_data, test_data)

#wykres procentowy w zależnosci od k dla wszytskich 4 cech naraz
plt.bar(range(1,16), result["percentages"], color='blue', label="Procent")
plt.ylim(70, 102)
plt.xticks(range(1, 16))
plt.title("Sumaryczny wynik klasyfikacji w zależności od k")
plt.xlabel("k (liczba sąsiadów)")
plt.ylabel("Procent [%]")
plt.savefig(("all_four"+".png"), dpi=300)
#plt.show()

print("wyniki dla wszystkich 4 cech naraz")
best_outcome = result["best outcome"]
print("najlepszy sumaryczny wynik klasyfikacji osiągnięto dla k =", result["best k"], " i wynosił on", f"{(result["best percentage"]):.2f}", "%")
print("Tak prezentuje się matryca błędów dla powyższego k")
print(best_outcome["confusion matrix"])
print("")

algorithm.print_data_and_draw(train_data, test_data, "sepal_length", "sepal_width")

algorithm.print_data_and_draw(train_data, test_data, "sepal_length", "petal_width")

algorithm.print_data_and_draw(train_data, test_data, "sepal_length", "petal_length")

algorithm.print_data_and_draw(train_data, test_data, "sepal_width", "petal_width")

algorithm.print_data_and_draw(train_data, test_data, "sepal_width", "petal_length")

algorithm.print_data_and_draw(train_data, test_data, "petal_width", "petal_length")