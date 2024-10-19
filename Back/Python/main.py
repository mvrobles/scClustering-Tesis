# uvicorn main:app --reload --port 8080
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import h5py
import scanpy as sc
import numpy as np
import os
import sys

from NeuralNetwork.NN_run_GMM import run_GMM
from read_data import *
from sc_experiment import *
from generate_outputs import *
from GraphBased.kMST import *
from NaiveBayes.main_naive_bayes import run_naive_bayes
from dict_tissues import tejidos_dict
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

single_cell_experiment = None

@app.post("/lecturaDatos")
async def cargar_analizar_datos(file: UploadFile = File(...)):
    """
    Endpoint to load and analyze data.

    Parameters:
    - file (UploadFile): The file to be loaded and analyzed.

    Returns:
    - dict: A dictionary containing the number of genes and number of cells in the loaded data.
    """
    print(file)
    x, _ = read_data(file.file)
    num_celulas = x.shape[0]
    num_genes = x.shape[1]
    print(num_celulas, num_genes)
    return {"num_genes": num_genes, "num_celulas": num_celulas}

@app.post("/lecturaArchivos")
async def cargar_datos(file_mtx: UploadFile = File(...),
                       file_barcodes: UploadFile = File(...),
                       file_genes: UploadFile = File(...)):
    """
    Loads data from uploaded files and performs necessary operations.

    Parameters:
    - file_mtx (UploadFile): The uploaded file containing the matrix data.
    - file_barcodes (UploadFile): The uploaded file containing the barcodes data.
    - file_genes (UploadFile): The uploaded file containing the genes data.

    Returns:
    - dict: A dictionary containing the number of genes and cells in the loaded data.
    """
    try:
        folder_path = '../upload_temp/'
        print("Comienza la lectura de los archivoss")
        path_mtx = download_to_local(file_mtx, folder_path)
        path_barcodes = download_to_local(file_barcodes, folder_path)
        path_features = download_to_local(file_genes, folder_path)
        
        x = initialize_sc_experiment(path_mtx, path_barcodes, path_features)
        single_cell_experiment.verify_data()

        print("LEYO CORRECTAMENTE EL SINGLE CELL EXPERIMENT")
        print(single_cell_experiment)

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Error al leer el archivo"}, message ={e})
    
    return {"num_genes": x.shape[0], "num_celulas": x.shape[1], "status": 200}

def initialize_sc_experiment(path_mtx, path_barcodes, path_features):
    barcodes, genes, x = read_data_scexperiment(path_mtx, path_barcodes, path_features)

    anndata_p = sc.AnnData(x)
    anndata_p.obs = barcodes
    anndata_p.var = genes

    global single_cell_experiment 
    single_cell_experiment = ScExperiment(matrix=x, barcodes=barcodes, genes=genes)
    return x

def read_data_scexperiment(path_mtx, path_barcodes, path_features):
    if 'gz' in path_barcodes:
      barcodes = read_tsv_gz(path_barcodes)
    else:
      barcodes = read_tsv(path_barcodes)
    print('Leyó barcodes', barcodes.shape)

    if 'gz'in path_features:
      genes = read_tsv_gz(path_features)
    else:
       genes = read_tsv(path_features)
    print('Leyó genes', genes.shape)

    if 'gz' in path_mtx:
      x = read_mtx_gz(path_mtx).T
    else:
      x = read_mtx(path_mtx).T
    print('Leyó matriz', x.shape)
    return barcodes,genes,x
    
@app.post("/CorrerModeloGrafos")
async def correr_kmst():
    """
    Runs the KMST model on a single cell experiment.

    Returns:
      dict: A dictionary with a message indicating whether the model was run successfully or not.
    """
    try:
        start_time = time.time()
        results_path = '../upload_temp/results/'
        
        print("ENTRO A CORRER KMST", single_cell_experiment)
        clusters = run_kmst(X = single_cell_experiment.matrix, 
                         barcodes = single_cell_experiment.barcodes, 
                         path_results = results_path, 
                         filter = 'mean-variance')
        end_time = time.time()
        generate_outputs(x = single_cell_experiment.matrix, 
                         clusters = clusters.cluster, 
                         output_path = results_path)
        execution_time = round(end_time - start_time,2)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Error al correr el modelo"}, message ={e})
    
    return {"message": "Modelo corrido correctamente", "time": execution_time}

@app.post("/CorrerModeloGMM")
async def correr_nn_gmm(request: Request):
  """
  Runs the NN model on a single cell experiment.

  Returns:
    dict: A dictionary with a message indicating whether the model was run successfully or not.
  """
  try:
      data = await request.json()
      n_clusters = int(data['n_clusters'])
      single_cell_experiment.n_clusters = n_clusters
      
      start_time = time.time()
      clusters, distr = run_GMM(X = single_cell_experiment.matrix, 
                barcodes = single_cell_experiment.barcodes,
                path_results = '../upload_temp/results/',
                n_clusters = single_cell_experiment.n_clusters)
      end_time = time.time()
      generate_outputs(x = single_cell_experiment.matrix, 
                        clusters = clusters.cluster, 
                        output_path = '../upload_temp/results/')
      generate_output_gmm(prob = distr, output_path = '../upload_temp/results/')
      execution_time = round(end_time - start_time,2)
  except Exception as e:
      return JSONResponse(status_code=400, content={"message": "Error al correr el modelo"}, message ={e})
  
  return {"message": "Modelo corrido correctamente", "time": execution_time}

@app.post("/CorrerModeloNB")
async def correr_nb(request: Request):
  """
  Runs the Naive Bayes model on a single cell experiment.

  Returns:
    dict: A dictionary with a message indicating whether the model was run successfully or not.
  """
  try:        
      data = await request.json()
      tissue = data['tissue_spanish']
      tissue = tejidos_dict[tissue]

      single_cell_experiment.tissue = tissue
      
      start_time = time.time()
      clusters = run_naive_bayes(X = single_cell_experiment.matrix, 
             barcodes = single_cell_experiment.barcodes,
             features = single_cell_experiment.genes,
             path_results = '../upload_temp/results/',
             tissue = single_cell_experiment.tissue)
      
      end_time = time.time()
      generate_output_nb(x = single_cell_experiment.matrix,
                         barcodes_clusters = clusters, 
                         real_tissue = single_cell_experiment.tissue,
                        output_path = '../upload_temp/results/')
      execution_time = round(end_time - start_time,2)
  except Exception as e:
      return JSONResponse(status_code=400, content={"message": "Error al correr el modelo"}, message ={e})
  
  return {"message": "Modelo corrido correctamente", "time": execution_time}
   

def read_data(file):
  data_mat = h5py.File(file)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y
