import numpy as np
from scipy.special import gammaln
from tqdm import tqdm

class NaiveBayes:
    def __init__(self, label_encoder, alpha=1.0):
        self.alpha = alpha # Suavizado de laplace
        self.label_encoder = label_encoder
    
    def compute_priors(self, cell_types, tissue):
        cell_types = list(set(cell_types))
        columnas_tissue = [c for c in cell_types if tissue in c]
        columnas_no_tissue = [c for c in  cell_types if c not in columnas_tissue]

        priors_tissue = {c: 0.9/len(columnas_tissue) for c in columnas_tissue}
        priors_no_tissue = {c: 0.1/len(columnas_no_tissue) for c in columnas_no_tissue}

        priors = {**priors_tissue, **priors_no_tissue}
        priors = {self.label_encoder.transform([str(key)])[0]: value for key, value in priors.items()}
        priors = dict(sorted(priors.items()))
        priors = np.array(list(priors.values()))

        self.class_priors = priors

    def fit(self, X, y):
        # Entrenar modelo
        pass

    def predict_log_proba(self, X):
        # Calcular logaritmo de la probabilidad a posteriori
        pass

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.cls_labels[np.argmax(log_probs, axis=1)]

class CustomMultinomial(NaiveBayes):
    def fit(self, X, y):
        X_smoothed = X + self.alpha
        _, n_features = X.shape
        self.cls_labels = np.unique(y)
        n_classes = len(self.cls_labels)
        self.feature_probs = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.cls_labels):
            X_cls = X_smoothed[y == cls] 
            self.feature_probs[idx, :] = np.sum(X_cls, axis=0)/ (np.sum(X_cls))

    def predict_log_proba(self, X):
        log_probs = np.log(self.class_priors) + X @ np.log(self.feature_probs.T)
        return log_probs

class NaiveBayesPoisson(NaiveBayes):
    def fit(self, X, y):
        self.cls_labels = np.unique(y)
        X_smoothed = X + self.alpha
        _, n_features = X.shape
        n_classes = len(self.cls_labels)
        
        # Inicializamos los parámetros
        self.mu = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.cls_labels):
            X_cls = X_smoothed[y == cls]  # Subconjunto de muestras de la clase actual
            self.mu[idx, :] = np.mean(X_cls, axis=0)

    def _poisson_logpmf(self, X, mu):
        return X * np.log(mu) - mu - gammaln(X + 1)

    def predict_log_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.cls_labels)))

        for idx, _ in enumerate(self.cls_labels):
            log_prior = np.log(self.class_priors[idx])  # Logaritmo de la probabilidad previa
            log_likelihood = np.sum(self._poisson_logpmf(X, self.mu[idx, :]), axis=1)  # Logaritmo de la verosimilitud
            log_probs[:, idx] = log_prior + log_likelihood
        
        return log_probs
    

    import numpy as np

class NaiveBayesNegBinomial(NaiveBayes):
    def __init__(self, label_encoder, alpha=1.0, theta_default=1.0):
        self.alpha = alpha 
        self.theta_default = theta_default  
        self.cls_labels = None
        self.mu = None
        self.theta = None
        self.label_encoder = label_encoder
    
    def fit(self, X, y):
        self.cls_labels = np.unique(y)
        _, n_features = X.shape
        X_smoothed = X + self.alpha

        n_classes = len(self.cls_labels)
        
        # Inicializamos los parámetros
        self.mu = np.zeros((n_classes, n_features))
        self.theta = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.cls_labels):
            X_cls = X_smoothed[y == cls]  
            self.mu[idx, :] = np.mean(X_cls, axis=0)

            if X_cls.shape[0] == 1:
                self.theta[idx, :] = self.theta_default
            else:
                var = np.var(X_cls, axis=0)
                self.theta[idx, :] = np.where(var > self.mu[idx, :], 
                                              (self.mu[idx, :]**2) / (var - self.mu[idx, :]), 
                                              self.theta_default)

    
    def _neg_binomial_logpmf(self, X, mu, theta):
        return (gammaln(X + 1/theta) - gammaln(1/theta) - gammaln(X + 1) + 
                X * np.log(mu/(mu + theta)) + (1/theta) * np.log(theta/(mu + theta)))
    
    def predict_log_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.cls_labels)))

        for idx, _ in enumerate(self.cls_labels):
            log_prior = np.log(self.class_priors[idx])  # Logaritmo de la probabilidad previa
            log_likelihood = np.sum(self._neg_binomial_logpmf(X, self.mu[idx, :], self.theta[idx, :]), axis=1)
            log_probs[:, idx] = log_prior + log_likelihood
        
        return log_probs
