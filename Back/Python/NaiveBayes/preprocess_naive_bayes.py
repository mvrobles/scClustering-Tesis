import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Datasets:
    def __init__(self, hpa, sc) -> None:
        self.hpa = hpa
        self.sc = sc
        self.label_encoder = None

    def filter_normalize(self):
        intersection_genes = list(set(self.hpa.genes).intersection(set(self.sc.genes)))
        self.hpa.filter_genes(intersection_genes)
        self.sc.filter_genes(intersection_genes)

        mediana_conteos = self.sc.normalize()
        self.hpa.normalize(mediana_conteos)

        genes_seleccionados = self.sc.select_genes()
        self.hpa.filter_genes(genes_seleccionados)

    def get_values_train_df(self, df):
        X = df.values
        y_text = df.index.droplevel(2).values
        y_text = np.array([str(x) for x in y_text])
        return X, y_text

    def get_values(self):
        X_hpa, y_text_hpa = self.get_values_train_df(self.hpa.df)
        X_sc = self.sc.df.values

        le = LabelEncoder()
        y_hpa = le.fit_transform(y_text_hpa)

        self.label_encoder = le
        return X_hpa, X_sc, y_hpa, y_text_hpa

class HPADataset:
    def __init__(self, path) -> None:
        self.df = pd.read_csv(path, sep = '\t')
        self.genes = self.df['Gene name'].values
        self.process()

    def process(self):
        self.df = self.df.pivot_table(
            index=['Gene name'], 
            columns=['Tissue', 'Cell type', 'Cluster'], 
            values='Read count').T

    def filter_genes(self, selected_genes):
        self.df = self.df[selected_genes]
        self.genes = self.df.index.values

    def normalize(self, mediana_conteos):
        for c in self.df.index:
            total = self.df.loc[c].sum()
            self.df.loc[c] = np.log1p(self.df.loc[c] * mediana_conteos / total)
    
class ScDataset:
    def __init__(self, X, barcodes, features) -> None:
        self.df = self.read(X, barcodes, features)
        self.genes = list(self.df.columns)

    def read(self, X, barcodes, features):
        try:
            genes_hl = list(features[features.columns[1]].values)
        except:
            genes_hl = list(features[features.columns[0]].values)

        cells_hl = list(barcodes[0].values)
        return pd.DataFrame(X, columns = genes_hl, index = cells_hl)
    
    def filter_genes(self, selected_genes):
        self.df = self.df[selected_genes]
        self.genes = list(self.df.columns)

    def select_genes(self):
        self.df = self.df.loc[:,~self.df.columns.duplicated()]
        num_genes = len(self.genes)
        genes_varianza = self.df.var(axis = 0).sort_values(ascending=False)
        genes_mayor_varianza = genes_varianza.index[0:int(0.1*num_genes)]
        self.df = self.df[genes_mayor_varianza]
        print('tamaño selfdf', self.df.shape, len(genes_mayor_varianza))

        return genes_mayor_varianza

    def normalize(self):
        mediana_conteos = self.df.sum(axis=1).median()

        df_normalized = self.df.copy()
        for c in self.df.index:
            total = self.df.loc[c].sum()
            df_normalized.loc[c] = np.log1p(self.df.loc[c] * mediana_conteos / total)
        self.df = df_normalized

        return mediana_conteos

