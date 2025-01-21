import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

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
            #sprawdzamy czy więcej nz jedna grupa ma maxymalna ilosc głosów czyli czy jest remis czy nie
            for i in range(len(group)):
                if group[i] == maximum:
                    is_only += 1

            #jeżeli jest remis usuwamy ostatni element z listy najbliższych punktów i powtarzamy zliczanie głosów
            if is_only!=1:
                closest_points.pop()
            else:
                #Jeżeli nie ma remisu czyli wynik jest jednoznaczny sprawdzamy która grupa wygrała i taki "result" zwracamy
                #oraz kończymy grupowanie dla tego punktu
                for i in range(len(group)):
                    if group[i] == maximum:
                        result = i
                predictions.append(result)
                not_done = False

    #Zwracamy słownik z wynikami(procent sukcesu, matryca błędów)
    return confusion_matrix(test_data.species, predictions)



def euclidean_distance(value1, value2):
    temp = 0
    if len(value1) != len(value2):
        return float('inf')
    for i in range(len(value1)-1):
        temp += (value1[i] - value2[i]) ** 2
    return sqrt(temp)

def confusion_matrix(real, predicted):
    matrix = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    correct = 0

    #iterujemy równoczesnie przez prawdziwe grupy do którech należały punkty oraz te "przewidywane" przez algorytm
    for real_group, predicted_group in zip(real, predicted):
        #zarówno wartosci real jak i predict są w zakresie 0-2 więc można uznać je za indeksy i dodać w to miejsce jeden
        matrix[real_group][predicted_group] += 1
        #co więcej jeżeli real i predicted są równe to oznacza że algorytm poprawnie przewidział grupę
        #dodajemy więc jeden do poprawnych wyników
        if real_group == predicted_group:
            correct += 1

    #Wyliczamy procent udanych przediwywań
    percentage = correct/(len(real))
    #zwracamy słownik z naszymi wynikami
    return {"confusion matrix": matrix, "percentage": percentage}

def repeat(train_data, test_data):
    outcomes = []
    percentages = []
    best_percentage = 0
    best_k = 0
    best_outcome = {}

    for k in range(1, 16):
        # powtarzamy algorytm 15 razy dla kolejnych k
        outcomes.append(custom_knn(k, train_data, test_data))
        #osiągnięty wynik procentowy zapisujemy
        percentage = float(outcomes[k - 1]["percentage"])

        #a następnie porównujemy z poprzednim aby znaleźc najwyższy procent czyli defakto najlepszy rezultat algorytmu
        if percentage > best_percentage:
            #w przypadku znalezienia nowego maximum(procentowego) zapisujemy zarówno ten procent
            #jak i k dla którego został osiągnięty oraz cały słownik z wynikiem
            best_percentage = percentage
            best_k = k
            best_outcome = outcomes[k - 1]

    for obj in outcomes:
        #przeliczamy ułamki na procenty żeby na wykresie były jako takie
        percentages.append(obj["percentage"] * 100)

    best_percentage = best_percentage * 100

    #zwracamy słownik z wynikami
    return {"outcomes": outcomes, "percentages": percentages, "best outcome": best_outcome, "best percentage": best_percentage, "best k": best_k}


def print_data_and_draw(train_data, test_data, kolumn1, kolumn2, percentage):
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
    #Zaczynamy od 70% bo wykresy lepiej wyglądają(słowa Nowaka nie moje) a najniższy wynik procentowy to chyba 72%
    #a kończymy na 102 bo przy 100 to srednio wygladało gdy wynik procentowy był równy 100 bo nie wiadomo było
    #czy to faktycznie już koniec czy te procenty idą w górę jeszcze tylko wykres się skończył
    plt.ylim(percentage, 102)
    plt.xticks(range(1, 16))
    plt.title("Sumaryczny wynik klasyfikacji w zależności od k")
    plt.xlabel("k (liczba sąsiadów)")
    plt.ylabel("Procent [%]")
    plt.savefig((kolumn1 + "_" + kolumn2 + ".png"), dpi=300)
    plt.show()

    print("wyniki dla", kolumn1, "i", kolumn2)
    best_outcome = result["best outcome"]
    print("najlepszy sumaryczny wynik klasyfikacji osiągnięto dla k =", result["best k"], " i wynosił on", f"{(result["best percentage"]):.2f}", "%")
    print("Tak prezentuje się matryca błędów dla powyższego k")
    print(best_outcome["confusion matrix"])
    print("")

