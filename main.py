import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def single_linkage_clustering(dist_matrix):
    n = len(dist_matrix)
    clusters = [[i] for i in range(n)]
    history = []
    k = 1
    file = open("result.txt", "w")
    file.write("-------Estrategia de distancia mínima-------\n")
    while len(clusters) > 1:
        min_dist = np.inf
        merge_indices = ()

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        if dist_matrix[idx1, idx2] < min_dist:
                            min_dist = dist_matrix[idx1, idx2]
                            merge_indices = (i, j)

        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        history.append((clusters[merge_indices[0]], clusters[merge_indices[1]], min_dist))
        del clusters[merge_indices[1]]
        clusters[merge_indices[0]] = merged_cluster

        for i in range(n):
            if i != merge_indices[0]:
                for idx in merged_cluster:
                    dist_matrix[i, idx] = min(
                        dist_matrix[i, idx], dist_matrix[i, clusters[merge_indices[0]][0]])
                dist_matrix[idx, i] = dist_matrix[i, idx]

        file.write(f"Paso {k}\n")
        file.write(f"Unión de clusters {merge_indices} (Distancia: {min_dist})\n")
        file.write(f"Clusters actuales: {clusters}\n")
        file.write("Matriz de distancia actualizada\n")
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                file.write(f"{dist_matrix[i][j]} \t")
            file.write("\n")
        file.write("\n")
        k += 1
    file.close()

    return clusters[0], history

def complete_linkage_clustering(dist_matrix):
    n = len(dist_matrix)
    clusters = [[i] for i in range(n)]
    history = []
    k = 1
    file = open("result.txt", "a")
    file.write("\n")
    file.write("-------Estrategia de similitud máxima-------\n")
    while len(clusters) > 1:
        min_dist = np.inf
        merge_indices = ()

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        if dist_matrix[idx1, idx2] < min_dist:
                            min_dist = dist_matrix[idx1, idx2]
                            merge_indices = (i, j)

        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        history.append((clusters[merge_indices[0]], clusters[merge_indices[1]], min_dist))
        del clusters[merge_indices[1]]
        clusters[merge_indices[0]] = merged_cluster

        for i in range(n):
            if i != merge_indices[0]:
                for idx in merged_cluster:
                    dist_matrix[i, idx] = max(
                        dist_matrix[i, idx], dist_matrix[i, clusters[merge_indices[0]][0]])
                dist_matrix[idx, i] = dist_matrix[i, idx]

        file.write(f"Paso {k}\n")
        file.write(f"Unión de clusters {merge_indices} (Distancia: {min_dist})\n")
        file.write(f"Clusters actuales: {clusters}\n")
        file.write("Matriz de distancia actualizada\n")
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                file.write(f"{dist_matrix[i][j]} \t")
            file.write("\n")
        file.write("\n")
        k += 1
    file.close()

    return clusters[0], history

def average_linkage_clustering(dist_matrix):
    n = len(dist_matrix)
    clusters = [[i] for i in range(n)]
    history = []
    k = 0
    file = open("result.txt", "a")
    file.write("\n")
    file.write("-------Estrategia de distancia media-------\n")
    while len(clusters) > 1:
        avg_dist = np.inf
        merge_indices = ()

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dists = [dist_matrix[idx1, idx2] for idx1 in clusters[i] for idx2 in clusters[j]]
                avg_dist_ij = sum(dists) / len(dists)
                if avg_dist_ij < avg_dist:
                    avg_dist = avg_dist_ij
                    merge_indices = (i, j)

        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        history.append((clusters[merge_indices[0]], clusters[merge_indices[1]], avg_dist))
        del clusters[merge_indices[1]]
        clusters[merge_indices[0]] = merged_cluster

        for i in range(n):
            if i != merge_indices[0]:
                dists = [dist_matrix[i, idx] for idx in merged_cluster]
                avg_dist_new = sum(dists) / len(dists)
                for idx in merged_cluster:
                    dist_matrix[i, idx] = avg_dist_new
                dist_matrix[idx, i] = avg_dist_new

        file.write(f"Paso {k}\n")
        file.write(f"Unión de clusters {merge_indices} (Distancia: {avg_dist})\n")
        file.write(f"Clusters actuales: {clusters}\n")
        file.write("Matriz de distancia actualizada\n")
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                file.write(f"{round(dist_matrix[i][j], 2)} \t")
            file.write("\n")
        file.write("\n")
        k += 1
    file.close()

    return clusters[0], history

distance_matrix = np.array([
    [0, 2.15, 0.7, 1.07, 0.85, 1.16, 1.56],
    [2.15, 0, 1.53, 1.14, 1.38, 1.01, 2.83],
    [0.7, 1.53, 0, 0.43, 0.21, 0.55, 1.86],
    [1.07, 1.14, 0.43, 0, 0.29, 0.22, 2.04],
    [0.85, 1.38, 0.21, 0.29, 0, 0.41, 2.02],
    [1.16, 1.01, 0.55, 0.22, 0.41, 0, 2.05],
    [1.56, 2.83, 1.86, 2.04, 2.02, 2.05, 0]
])

def plot_dendrogram(history, title, fig_name):
    Z = []
    cluster_map = {i: i for i in range(len(history) + 1)}
    next_cluster_id = len(history) + 1

    for cluster1, cluster2, dist in history:
        new_cluster = next_cluster_id
        next_cluster_id += 1

        c1_id = cluster_map[cluster1[0]]
        c2_id = cluster_map[cluster2[0]]

        Z.append([c1_id, c2_id, dist, len(cluster1) + len(cluster2)])
        for idx in cluster1 + cluster2:
            cluster_map[idx] = new_cluster

    Z = np.array(Z)
    dendrogram(Z)
    plt.title(title)
    plt.savefig(fig_name)
    plt.show()

_, history = single_linkage_clustering(distance_matrix.copy())
plot_dendrogram(history, "Agrupamiento por Enlace Simple", "Distancia_minima2")

_, history = complete_linkage_clustering(distance_matrix.copy())
plot_dendrogram(history, "Agrupamiento por Enlace Completo", "Similitud_maxima2")

_, history = average_linkage_clustering(distance_matrix.copy())
plot_dendrogram(history, "Agrupamiento por Enlace Promedio", "Distancia_promedio2")