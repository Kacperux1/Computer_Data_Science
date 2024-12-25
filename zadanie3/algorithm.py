import pandas as pd
import matplotlib.pyplot as plt

def custom_knn(k, train_data, test_data):
    predictions = []

    #przechodzimy przez wszystkie obiekty ze zbioru testowego aby znaleźć im dopasowanie
    for new_obj in test_data.values:
        closest_points = []
        distances = []

        #wyliczamy odleglosc obiektu ze zbioru testowego do kazdego obiektu ze zbioru treningowego i zapisujemy jego index
        for i, obj in enumerate(train_data.values):  #values konwertuje wiersze na tablice NumPy
            distances.append([euclidean_distance(obj, new_obj), i])

        #sortujemy liste
        distances.sort()

        #z posortowanej listy wybieramy pierwsze tyle punktów dla ilu k jest algorytm
        for i in range(k):
            closest_points.append(train_data.iloc[distances[i][1]].to_dict())

        #pętla while bedzie dizłała tak długo aż osiągniemy jednoznaczny wynik przyporządkowania do klasy
        not_done=True
        while not_done:
            group = [0, 0, 0]
            #zliczamy ile "głosów" jest dla poszczególnych klas
            for obj in closest_points:
                if obj["species"] == 0:
                    group[0] += 1
                elif obj["species"] == 1:
                    group[1] += 1
                else:
                    group[2] += 1
            #sprawdzamy
            maximum = max(group)
            is_only = 0
            result = 0
            #sprawdzamy czy grupa
            for i in range(len(group)):
                if group[i] == maximum:
                    is_only += 1

            if is_only!=1:
                closest_points.pop()
            else:
                for i in range(len(group)):
                    if group[i] == maximum:
                        result = i
                predictions.append(result)
                not_done = False


    matrix_percentage = confusion_matrix(test_data.species, predictions)
    matrix_percentage["predictions"] = predictions

    return matrix_percentage



def euclidean_distance(value1, value2):
    temp = 0
    if len(value1) != len(value2):
        return float('inf')
    for i in range(len(value1)-1):
        temp += (value1[i] - value2[i]) ** 2
    return temp

def confusion_matrix(real, predicted):
    matrix = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    correct = 0

    for real_group, predicted_group in zip(real, predicted):
        matrix[real_group][predicted_group] += 1
        if real_group == predicted_group:
            correct += 1

    percentage = correct/(len(real))
    return {"confusion matrix": matrix, "percentage": percentage}

def repeat(train_data, test_data):
    outcomes = []
    percentages = []
    best_percentage = 0
    best_k = 0
    best_outcome = {}
    for i in range(1, 16):
        outcomes.append(custom_knn(i, train_data, test_data))
        percentage = float(outcomes[i - 1]["percentage"])

        if percentage > best_percentage:
            best_percentage = percentage
            best_k = i
            best_outcome = outcomes[i - 1]

    for obj in outcomes:
        percentages.append(obj["percentage"] * 100)

    return {"outcomes": outcomes, "percentages": percentages, "best outcome": best_outcome, "best percentage": best_percentage, "best k": best_k}

def print_data_and_draw(train_data, test_data, kolumn1, kolumn2):
    # robimy nowe train_data zawierające tylko dwie cechy i gatunek naraz
    train_data_dual = train_data[[kolumn1, kolumn2, "species"]].values
    columns = [kolumn1, kolumn2, "species"]
    train_data_dual = pd.DataFrame(train_data_dual, columns=columns)
    # musimy zamienić gatunek na int bo się kopiował jako float
    train_data_dual["species"] = train_data_dual["species"].astype(int)

    # To samo co dla train_data robimy dla  test_data
    test_data_dual = test_data[[kolumn1, kolumn2, "species"]].values
    columns = [kolumn1, kolumn2, "species"]
    test_data_dual = pd.DataFrame(test_data_dual, columns=columns)
    test_data_dual["species"] = test_data_dual["species"].astype(int)

    # wywołujemy nasz algorytm
    result = repeat(train_data_dual, test_data_dual)

    # Robimy wykres procenwó w zaleznosci od k dla dwoch cech
    plt.bar(range(1, 16), result["percentages"], color='blue', label="Procent")
    plt.ylim(70, 102)
    plt.xticks(range(1, 16))
    plt.title("Sumaryczny wynik klasyfikacji w zależności od k")
    plt.xlabel("k (liczba sąsiadów)")
    plt.ylabel("Procent [%]")
    plt.savefig((kolumn1 + "_" + kolumn2 + ".png"), dpi=300)
    plt.show()

    print("wyniki dla", kolumn1, "i", kolumn2)
    best_outcome = result["best outcome"]
    print("najlepszy sumaryczny wynik klasyfikacji osiągnięto dla k =", result["best k"], " i wynosił on", f"{(result["best percentage"]*100):.2f}", "%")
    print("Tak prezentuje się matryca błędów dla powyższego k")
    print(best_outcome["confusion matrix"])
    print("")

