import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 18})

def reduce_dimension(x: np.array) -> np.array:
    """
    Reduces the dimensionality of the input array using t-SNE algorithm.

    Parameters:
    - x (np.array): Input array to be dimensionally reduced.

    Returns:
    - np.array: Dimensionally reduced array.

    """

    x_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3, random_state = 123).fit_transform(x)
    return x_embedded

def paint_tsne(x: np.array, 
               output_path: str, 
               clusters: list) -> None:
    """
    Generate a scatterplot of the t-SNE reduced dimensions of the input data.

    Parameters:
    - x (np.array): The input data array.
    - output_path (str): The path where the output image will be saved.
    - clusters (list): The list of cluster labels for each data point.

    Returns:
    - None
    """
    adata = sc.AnnData(x)
    adata.obs['cluster'] = clusters.astype(str).values
    num_clusters = clusters.nunique()

    plt.figure(figsize=(8,6))
    palette = sns.color_palette("tab20", num_clusters)
    sc.tl.tsne(adata, use_rep='X', n_pcs=0, perplexity=30)
    sc.pl.tsne(adata, color='cluster', title='t-SNE agrupación', size=50, show=False, palette=palette)
    
    plt.savefig(output_path + 'tsne_clusters.png')
    plt.close()

def paint_cluster_distributions(clusters: np.array,
                                output_path: str) -> None:
    """
    Paints the cluster distributions using a countplot and saves the plot as an image.

    Parameters:
    - clusters (np.array): An array containing the cluster labels.
    - output_path (str): The path where the image will be saved.

    Returns:
    - None
    """
    plt.figure(figsize=(8,6))
    sns.countplot(x=clusters, alpha = 0.8).set(
        title = 'Distribución de clusters',
        xlabel = 'Cluster',
        ylabel = 'Células',
    )
    plt.savefig(output_path + 'cluster_distributions.png')
    plt.close()

def generate_outputs(x: np.array,
                     clusters: np.array,
                     output_path: str) -> None:
    """
    Generate tsne plot and countplot for the given data.

    Parameters:
    - x (np.array): The input data.
    - clusters (np.array): The cluster assignments for each data point.
    - output_path (str): The path to save the generated outputs.
    """
    paint_tsne(x, output_path, clusters)
    paint_cluster_distributions(clusters, output_path)

def generate_output_gmm(prob: np.array, output_path: str) -> None:
    """
    Generate tsne plot and countplot for the given data.

    Parameters:
    - x (np.array): The input data.
    - clusters (np.array): The cluster assignments for each data point.
    - output_path (str): The path to save the generated outputs.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(prob, cmap = 'viridis').set(
        title = 'Distribución de probabilidad por célula',
    )
    plt.savefig(output_path + 'gmm_probabilities.png')
    plt.close()

def generate_output_nb(x: np.array, barcodes_clusters: np.array, 
                       output_path: str, real_tissue: str) -> None:
    plot_tissue_distribution(barcodes_clusters, output_path)
    plot_type_distribution(barcodes_clusters, output_path, real_tissue)

    label_encoder = LabelEncoder()
    barcodes_clusters['num_cluster'] = label_encoder.fit_transform(barcodes_clusters.cluster.values)
    paint_tsne(x, output_path, barcodes_clusters.num_cluster)
    
def plot_type_distribution(barcodes_clusters, output_path, real_tissue):
    barcodes_tissue = barcodes_clusters[barcodes_clusters.tissue == real_tissue]
    plt.figure()
    b= sns.barplot(barcodes_tissue.type.value_counts(), orient='h')
    b.set(
        title = "Distribución de tipos celulares \n Tejido: " + real_tissue,
        ylabel = "",
        xlabel = "Número de células asignadas",
    )

    for container in b.containers:
        b.bar_label(container, label_type='edge', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path + 'nb_type_distribution.png')

def plot_tissue_distribution(barcodes_clusters, output_path):
    plt.figure()
    b= sns.barplot(barcodes_clusters.tissue.value_counts(), orient='h')
    b.set(
        title = "Distribución de tejidos asignados",
        ylabel = "",
        xlabel = "Número de células asignadas",
    )

    for container in b.containers:
        b.bar_label(container, label_type='edge', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path + 'nb_tissue_distribution.png')
