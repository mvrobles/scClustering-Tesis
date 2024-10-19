import numpy as np
import pandas as pd 

from .preprocess_naive_bayes import HPADataset, ScDataset, Datasets
from .naive_bayes import NaiveBayesPoisson, CustomMultinomial

def run_naive_bayes(X: np.array, 
             barcodes: pd.DataFrame, 
             features: pd.DataFrame,
             path_results: str, 
             tissue: str,
             alpha: float = 0.001) -> pd.DataFrame:
    print('ENTRO A NAIVE BAYES')

    # Read HPA and scExperiment data
    hpa_dataset = HPADataset('../upload_temp/rna_single_cell_type_tissue.tsv')
    sc_dataset = ScDataset(X = X, barcodes = barcodes, features = features)

    print("Leyo los datasets", hpa_dataset.df.shape, sc_dataset.df.shape)
    # Normalize counts
    datasets = Datasets(hpa=hpa_dataset, sc = sc_dataset)
    datasets.filter_normalize()
    X_hpa, X_hl, y_hpa, y_hpa_text = datasets.get_values() 

    # Compute priors
    clf_nbp = CustomMultinomial(alpha = alpha, 
                                label_encoder = datasets.label_encoder)
    clf_nbp.compute_priors(y_hpa_text, tissue)

    # Train model
    clf_nbp.fit(X_hpa, y_hpa)

    print('Terminó el entrenamiento')

    # Predict
    y_pred_nbp = clf_nbp.predict(X_hl)
    y_pred_text = datasets.label_encoder.inverse_transform(y_pred_nbp)

    update_barcodes(barcodes, y_pred_text)

    barcodes.to_csv(path_results + 'nb_clusters.csv', index = False)
    print(f'-----> Se guardó correctamente el csv {path_results}')

    return barcodes

def update_barcodes(barcodes, y_pred_text):
    barcodes['cluster'] = y_pred_text

    barcodes['tissue'] = barcodes['cluster'].apply(
        lambda x: x.split(' ')[0].replace('(', '').replace("'", "").replace(',', ''))

    barcodes['type'] = barcodes['cluster'].apply(
        lambda x: x.split(' ')[1].replace(')', '').replace("'", "").replace(',', ''))

    barcodes['cluster'] = barcodes['cluster'].apply(
        lambda x: x.replace('(', '').replace("'", "").replace(',', '').replace(' ', '-').replace(')', ''))