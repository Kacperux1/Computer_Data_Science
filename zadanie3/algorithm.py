

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