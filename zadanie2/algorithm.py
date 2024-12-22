import random
from math import sqrt

from numpy import mean


class CustomKMeansResult:
    def __init__(self, cluster_number, groups, centers, iteration_number, wcss):
        self.cluster_number = cluster_number
        self.groups = groups
        self.centers = centers
        self.iteration_number = iteration_number
        self.wcss = wcss




def CustomKMeans(cluster_number, values, tolerance):
    results = []


    for repeat in range(10):
        clusters = [0] * len(values)
        iterations = 1
        finished = False

        new_centers = []
        centers = [random.choice(values) for i in range(cluster_number)]

        while not finished:

            for i in range(len(values)):
                distances = [euclidean_distance(values[i], centre) for centre in centers]
                clusters[i] = distances.index(min(distances))

            new_centers = []

            for i in range(len(centers)):

                cluster_points = [values[j] for j in range(len(values)) if clusters[j] == i]
                if cluster_points:

                    new_centers.append(list(mean(cluster_points, axis=0)))
                else:
                    new_centers.append(centers[i])

            if all(euclidean_distance(centers[i], new_centers[i]) < tolerance for i in range(len(centers))):
                finished = True
            else:
                iterations += 1
                centers = new_centers[:]

        results.append(CustomKMeansResult(len(centers), clusters, new_centers, iterations, wcss_calculate(values, clusters, centers)))

    minWCSS = min(obj.wcss for obj in results)

    for obj in results:
        if obj.wcss == minWCSS:
            return obj

def euclidean_distance(value1, value2):
    temp = 0
    if len(value1) != len(value2):
        return float('inf')
    for i in range(len(value1)):
        temp += (value1[i] - value2[i]) ** 2
    return sqrt(temp)

def wcss_calculate(values, clusters, centers):
    result = 0
    for i in range(len(centers)):
        for j in range(len(clusters)):
            if clusters[j] == i:
                result += euclidean_distance(values[j], centers[i]) ** 2
    return result
