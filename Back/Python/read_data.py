from scipy.sparse import csc_matrix
from scipy.io import mmread
import numpy as np
import pandas as pd
import shutil
import gzip
import os

def read_mtx_gz(file) -> csc_matrix:
    """
    Reads a .mtx file and returns a sparse matrix.
    """
    with gzip.open(file, 'rt') as f:
        return mmread(f).tocsc().toarray().astype(np.float32)

def download_to_local(file, local_folder) -> str:
    """
    Saves the uploaded file to the local folder.
    """
    local_file_path = os.path.join(local_folder, file.filename)
    with open(local_file_path, 'wb') as local_file:
        shutil.copyfileobj(file.file, local_file)
    return local_file_path

def read_tsv_gz(file):
    """
    Reads a .tsv.gz file and returns a DataFrame.
    """
    with gzip.open(file, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None)
    return df