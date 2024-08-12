""" Subnet Overlap Checker """

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import h5py
import numpy as np
import os
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # Lee el archivo .h5 usando h5py
#         with h5py.File(file.file, 'r') as f:
#             # Procesa el archivo h5, por ejemplo, obteniendo las claves
#             data = list(f.keys())
#             print(data)
#             response = JSONResponse(content={"message": "Archivo procesado", "data": data})
#             print(response)
#             return response

#     except Exception as e:
#         return JSONResponse(content={"message": "Error procesando el archivo", "error": str(e)}, status_code=400)

@app.post("/lecturaDatos")
async def cargar_analizar_datos(file: UploadFile = File(...)):

    x, _ = read_data(file.file)
    num_celulas = x.shape[0]
    num_genes = x.shape[1]
    return {"num_genes": num_genes, "num_celulas": num_celulas}


def read_data(file):
  data_mat = h5py.File(file)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y
