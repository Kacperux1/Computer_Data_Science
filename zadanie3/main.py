import pandas as pd
import algorithm

train_data = pd.read_csv("data3_train.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
test_data = pd.read_csv("data3_test.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

columns_to_convert = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
train_data[columns_to_convert] = train_data[columns_to_convert].astype(float)
test_data[columns_to_convert] = test_data[columns_to_convert].astype(float)

train_data["species"] = train_data["species"].astype(int)
test_data["species"] = test_data["species"].astype(int)

lista = algorithm.custom_knn(1, train_data, test_data)

print(lista)