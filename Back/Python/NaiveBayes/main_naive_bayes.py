import numpy as np
import pandas as pd 

from .preprocess_naive_bayes import HPADataset, ScDataset, Datasets
from .naive_bayes import NaiveBayesPoisson

def run_naive_bayes(X: np.array, 
             barcodes: pd.DataFrame, 
             features: pd.DataFrame,
             path_results: str, 
             tissue: str,
             alpha: float = 0.001) -> pd.DataFrame:
    # Read HPA and scExperiment data
    hpa_dataset = HPADataset('../../data/Datos - Human Protein Atlas/rna_single_cell_type_tissue.tsv')
    sc_dataset = ScDataset(X = X, barcodes = barcodes, features = features)

    # Normalize counts
    datasets = Datasets(hpa=hpa_dataset, sc = sc_dataset)
    datasets.filter_normalize()
    X_hpa, X_hl, y_hpa, y_hpa_text = datasets.get_values() 

    # Compute priors
    clf_nbp = NaiveBayesPoisson(alpha = alpha, 
                                            label_encoder = datasets.label_encoder)
    clf_nbp.compute_priors(y_hpa_text, tissue)

    # Train model
    clf_nbp.fit(X_hpa, y_hpa)

    # Predict
    y_pred_nbp = clf_nbp.predict(X_hl)
    barcodes['cluster'] = y_pred_nbp
    barcodes.to_csv(path_results + 'clusters.csv', index = False)
    print(f'-----> Se guardó correctamente el csv {path_results}')

    return barcodes