from scipy.sparse import csc_matrix

class ScExperiment:
    def __init__(self, 
                 matrix: csc_matrix,
                 barcodes: list,
                 genes: list,
                 n_clusters: int = None,
                 tissue: str = None) -> None:
        self.matrix = matrix
        self.barcodes = barcodes
        self.genes = genes
        self.n_clusters = n_clusters
        self.tissue = tissue

    def verify_data(self) -> bool:
        """
        Validates the data read from the file
        """
        assert self.matrix.shape[0] == len(self.barcodes), "Number of barcodes does not match the number of cells"
        assert self.matrix.shape[1] == len(self.genes), "Number of genes does not match the number of genes"
        assert len(self.barcodes) > 0, "No barcodes found"
        assert len(self.genes) > 0, "No genes found"
        if self.n_clusters is not None:
            assert self.n_clusters > 0, "Number of clusters must be greater than 0"
            assert self.n_clusters < len(self.barcodes), "Number of clusters must be less than the number of cells"

        return True
    
    def set_matrix(self, x: csc_matrix) -> None:
        self.matrix = x

    def set_barcodes(self, barcodes: list) -> None:
        self.barcodes = barcodes

    def set_genes(self, genes: list) -> None:
        self.genes = genes

    def set_tissue(self, tissue: str) -> None:
        self.tissue = tissue