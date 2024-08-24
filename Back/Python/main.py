from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import h5py
import scanpy as sc
import numpy as np
import os
import sys

from read_data import *
from sc_experiment import *
from kMST import *

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
        path_mtx = download_to_local(file_mtx, folder_path)
        path_barcodes = download_to_local(file_barcodes, folder_path)
        path_features = download_to_local(file_genes, folder_path)
        
        barcodes = read_tsv_gz(path_barcodes)
        print('Leyó barcodes', barcodes.shape)
        genes = read_tsv_gz(path_features)
        print('Leyó genes', genes.shape)
        x = read_mtx_gz(path_mtx).T
        print('Leyó matriz', x.shape)

        anndata_p = sc.AnnData(x)
        anndata_p.obs = barcodes
        anndata_p.var = genes

        global single_cell_experiment 
        single_cell_experiment = ScExperiment(matrix=x, barcodes=barcodes, genes=genes)
        single_cell_experiment.verify_data()

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Error al leer el archivo"}, message ={e})
    
    return {"num_genes": x.shape[0], "num_celulas": x.shape[1]}
    
@app.post("/CorrerModelo")
async def correr_kmst():
    """
    Runs the KMST model on a single cell experiment.
    
    Returns:
      dict: A dictionary with a message indicating whether the model was run successfully or not.
    """
    try:
        print('\n\nSINGLE CELL EXPERIMENT\n\n', single_cell_experiment)
        run_kmst(X = single_cell_experiment.matrix, 
                 barcodes = single_cell_experiment.barcodes, 
                 features = single_cell_experiment.genes, 
                 path_results = '../upload_temp/results/', 
                 filter = 'mean-variance')
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Error al correr el modelo"}, message ={e})
    
    return {"message": "Modelo corrido correctamente"}

def read_data(file):
  data_mat = h5py.File(file)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y
