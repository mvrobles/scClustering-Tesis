import argparse
import os
import pickle
import warnings
from scipy.optimize import curve_fit

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from tqdm import tqdm


warnings.filterwarnings('ignore')

def normalize(adata, filter_min_counts=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata.raw = adata
    sc.pp.normalize_per_cell(adata)

    if logtrans_input:
        sc.pp.log1p(adata)

    zero_percentage = np.where(adata.X==0,True,False).sum()/(adata.X.shape[0]*adata.X.shape[1])
    print('PORCENTAJE DE CEROS', zero_percentage)

    return adata

def filter_genes_variance(X):
    varianzas_columnas = np.var(X, axis=0)
    indices_mayor_varianza = np.argsort(varianzas_columnas)[::-1][:5000]
    X_filtered = X[:, indices_mayor_varianza]

    return X_filtered

def filter_genes_mean_variance(X):
    medias = np.mean(X, axis = 0)
    varianzas = np.var(X, axis = 0)

    def curve_func(x, a, n, b):
        return a * x / (x**n + b)

    popt, _ = curve_fit(curve_func, medias, varianzas)
    a,n,b = popt

    expected_variance = a * medias / (medias**n + b)
    observed_expected = varianzas - expected_variance
    indices_mayor_var = np.argsort(observed_expected)[::-1][:5000]
    X_filtered = X[:, indices_mayor_var]

    return X_filtered

def paint_matrix(correlaciones, output_path, name):
    plt.Figure()
    sns.heatmap(correlaciones)
    plt.savefig(output_path + name + '.png')


def get_correlations(X):
    correlaciones = np.corrcoef(X)
    
    #celulas = X.shape[0]
    # correlaciones = np.zeros((celulas, celulas))
    # for i in tqdm(range(celulas)):
    #     for j in range(i, celulas):
    #         corr = pearsonr(X[i, :], X[j, :])[0]
    #         correlaciones[i, j] = corr 
    #         correlaciones[j, i] = corr 

    return correlaciones

def save_matrix(correlaciones, output_path, name):
    with open(output_path + name + '.pickle', 'wb') as f:
        pickle.dump(correlaciones, f)


def create_kMST(distance_matrix, inverse = True, k = None, threshold = 1e-5):
    if k is None:
        N = np.log(len(distance_matrix))
        k = int(np.floor(N))
    
    print(f'k = {k}')
    grafo = nx.Graph()
    nodos = range(len(distance_matrix))

    # Crear nodo inicial
    grafo.add_nodes_from(nodos)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            peso = distance_matrix[i][j]
            if peso > threshold:
                # para MST necesito el inverso de las correlaciones
                if inverse:
                    grafo.add_edge(i, j, weight=1-peso)
                else:
                    grafo.add_edge(i, j, weight=peso)


    print(f'---> Number of edges: {grafo.number_of_edges()}')

    mst_antes = None
    # Creamos los MSTs
    for iter in tqdm(range(k)):
        mst_new = nx.minimum_spanning_tree(grafo)

        edges_to_remove = list(mst_new.edges)
        grafo.remove_edges_from(edges_to_remove)

        if mst_antes is None:
            mst_antes = mst_new.copy()
        else:
            mst_new.add_edges_from(list(mst_antes.edges()))
            mst_antes = mst_new.copy()

    return mst_antes 

def louvain(grafo):
    particiones = nx.community.louvain_communities(grafo, seed=123)

    diccionario = {}

    for i, conjunto in enumerate(particiones):
        for elemento in conjunto:
            diccionario[elemento] = i

    num_nodos = grafo.number_of_nodes()
    clusters = np.full(num_nodos, -1, dtype=int)

    for nodo, comunidad in diccionario.items():
        clusters[nodo] = comunidad

    return clusters

def cluster_acc_plot(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    
    plt.figure(figsize = (10,7))
    w_order = np.zeros((D, D), dtype=np.int64)
    for i in range(D):
        for j in range(D):
            w_order[i,j] = w[i, ind[1][j]]

    df_cm = pd.DataFrame(w_order, index = [i for i in range(D)], columns = [i for i in ind[1]])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.ylabel("Prediction")
    plt.xlabel("Ground Truth")
    plt.show()
    
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def run_kmst(X: np.array, 
             barcodes: pd.DataFrame, 
             features: pd.DataFrame, 
             path_results: str, 
             filter: str) -> None:
    # Converting to AnnData
    anndata_p = sc.AnnData(X)
    anndata_p.obs = barcodes
    anndata_p.var = features

    # Normalize 
    anndata_p = normalize(anndata_p)
    print(f"Se normalizaron correctamente los datos. Dimensiones: {anndata_p.X.shape}")
    
    X = anndata_p.X

    # Gene selection
    if filter == 'mean-variance':
        X = filter_genes_mean_variance(X)
        print(f"Se filtraron los genes. Dimensiones: {X.shape}")

    elif filter == 'variance':
        X = filter_genes_variance(X)
        print(f"Se filtraron los genes. Dimensiones: {X.shape}")

    # Correlaciones
    correlaciones = get_correlations(X)

    # Guardar datos
    paint_matrix(correlaciones, path_results, name = "correlaciones_heatmap")
    save_matrix(correlaciones,  path_results, name = "correlaciones")
    print('Se guardaron correctamente las correlaciones y el heatmap en la carpeta ' + path_results)

    # Crear y guardar MST
    kmst = create_kMST(distance_matrix = correlaciones, 
                       inverse = True, 
                       threshold = 0)
    
    print('-----> Terminó la creación del grafo kMST para k = logN')

    with open(path_results + 'kmst_graph.pickle', 'wb') as file:
        pickle.dump(kmst, file)

    print(f'-----> Se guardó correctamente el grafo kMST en {path_results}')

    # Clustering
    clusters = louvain(kmst)
    with open(path_results + 'clusers_kmst.pickle', "wb") as file:
        pickle.dump(clusters, file)
