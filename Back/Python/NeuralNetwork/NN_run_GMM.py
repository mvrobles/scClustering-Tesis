import argparse
import os
import pickle
import warnings
from time import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import Birch
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from Back.Python.NeuralNetwork.NN_GMM import scGMM

import sys
sys.path.append('../')

from Back.Python.NeuralNetwork.preprocess import normalize, read_dataset
from utils import cluster_acc

warnings.filterwarnings('ignore')

def set_hyperparameters():
    parameters = {
        'label_cells': 0.1,
        'label_cells_files': 'label_selected_cells_1.txt',
        'n_pairwise': 0,
        'n_pairwise_error': 0,
        'batch_size': 256,
        'maxiter': 100,
        'pretrain_epochs': 300,
        'gamma': 1.,
        'ml_weight': 1.,
        'cl_weight': 1.,
        'update_interval': 1,
        'tol': 0.001,
    }

    return parameters

def create_train_model(params, adata, n_clusters):
    # Create saving directory
    if not os.path.exists(params['path_results']):
        os.makedirs(params['path_results'])

    sd = 2.5

    # Model
    model = scGMM(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd,
                path = params['path_results'])#.cuda()

    print(str(model))

    # Training
    t0 = time()
    model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                            batch_size=params['batch_size'], epochs=params['pretrain_epochs'])

    print('Pretraining time: %d seconds.' % int(time() - t0))

    return model

def second_training(params, model, adata):
    t0 = time()
    
    # Second training: clustering loss + ZINB loss
    y_pred,  mu, pi, cov, z, epochs, clustering_metrics, losses = model.fit(X=adata.X, X_raw=adata.raw.X, 
                                    sf=adata.obs.size_factors, batch_size=params['batch_size'],  num_epochs=params['maxiter'],
                                    update_interval=params['update_interval'], tol=params['tol'], lr = 0.001, y = None)

    # Se guardan los resultados
    pd.DataFrame(z.cpu().detach().numpy()).to_csv(params['path_results'] + 'Z.csv', index = None)
    pd.DataFrame(mu.cpu().detach().numpy()).to_csv(params['path_results']  + 'Mu.csv', index = None)
    pd.DataFrame(pi.cpu().detach().numpy()).to_csv(params['path_results']  + 'Pi.csv', index = None)
    pd.DataFrame(cov.cpu().detach().numpy()).to_csv(params['path_results'] + 'DiagCov.csv', index = None)

    with open(params['path_results'] + '/prediccion.pickle', 'wb') as handle:
        pickle.dump(y_pred, handle)

    print('Time: %d seconds.' % int(time() - t0))

    return y_pred


def model_BIRCH(X: np.array, 
                n_clusters: int,
                threshold: list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                branching_factor: list =  [10, 50, 100, 150]
                ) -> tuple:
    """
    Trains a Birch model for the input data X by optimizing the hyperparameters threshold and branching factor.

    input:
    - X: array with data to cluster
    - n_clusters: number of clusters 
    - threshold: list of possible values of threshold for the Birch model
    - branching_factor: list of possible branching factor values for the Birch model

    output:
    - best_model: a Birch Model of sklearn that maximized the silhouette score for the data
    - params: tuple of 
    """
    # Hyperparameters to search
    param_grid = {
        'threshold': threshold,
        'branching_factor': branching_factor
    }

    max_sil = -2
    params = 0, 0 
    best_model = None 
    for t in param_grid['threshold']:
        for b in param_grid['branching_factor']:
            birch_model = Birch(n_clusters=n_clusters, threshold = t, branching_factor = b)
            birch_model.fit(X)

            labels = birch_model.predict(X)

            sil = silhouette_score(X, labels)
            if sil > max_sil:
                max_sil = sil
                params = t, b 
                best_model = birch_model

    return best_model, params, max_sil 

def unsupervised_metrics(X, y_pred):
    # Evaluación final de resultados: métricas comparando con los clusters reales
    sil = np.round(silhouette_score(X, y_pred), 5)
    chs = np.round(calinski_harabasz_score(X, y_pred), 5)
    dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    print('Evaluating cells: SIL= %.4f, CHS= %.4f, DBS= %.4f' % (sil, chs, dbs))

def run_GMM(X: np.array, 
            barcodes: pd.DataFrame,
            path_results: str, 
            n_clusters: int):
    params = set_hyperparameters()
    params['path_results'] = path_results

    # processing of scRNA-seq read counts matrix
    anndata_p = sc.AnnData(X)
    anndata_p.obs = barcodes

    # Normalize 
    anndata_p = normalize(anndata_p)
    barcodes = anndata_p.obs
    x = anndata_p.X 

    # Set k 
    n_clusters = int(n_clusters)

    # Model training
    model = create_train_model(params, anndata_p, n_clusters)   
    y_pred1 = second_training(params, model, anndata_p)

    print('----> Unsupervised metrics for GMM Autoencoder:')
    unsupervised_metrics(x, y_pred1)

    # Guardar resultados
    barcodes['cluster'] = y_pred1
    barcodes.to_csv(path_results + 'gmm_clusters.csv', index = False)
    print(f'-----> Se guardó correctamente el csv {path_results}')

    # # Birch
    # z = pd.read_csv(params['path_results'] + 'Z.csv').values
    # best_model, _, _ = model_BIRCH(X = z, n_clusters = n_clusters)
    # y_pred2 = best_model.predict(z)

    # # Guardar resultados
    # barcodes['cluster'] = y_pred2
    # barcodes.to_csv(path_results + 'gmm_birch_clusters.csv', index = False)
    # print(f'-----> Se guardó correctamente el csv {path_results}')
    
    # print('\n----> Unsupervised metrics for GMM Autoencoder + Birch:')
    # unsupervised_metrics(z, y_pred2)

    return barcodes