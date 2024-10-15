import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import scanpy as sc

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
    adata.obs['cluster'] = clusters.astype(str)

    #x_embedded = reduce_dimension(x)
    plt.figure(figsize=(8,6))
    sc.tl.tsne(adata, use_rep='X', n_pcs=0, perplexity=30)
    sc.pl.tsne(adata, color='cluster', title='t-SNE plot colored by cluster', size=50, show=False)
    #sns.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue = clusters.values, palette = "Set1", size = 1).set(
    #    title = 'Agrupación de células - Representación t-SNE',
    #    xlabel = 't-SNE_1',
    #    ylabel = 't-SNE_2',
    #    xticks = [],
    #    yticks = []
    #)
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