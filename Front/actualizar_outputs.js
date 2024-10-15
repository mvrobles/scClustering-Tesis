function actualizarImagenesGraph() {
    const timestamp = new Date().getTime(); // Para evitar caché, se agrega un timestamp único
    
    document.getElementById("ImagenCorrelaciones").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/correlaciones_heatmap.png?" + timestamp;
    
    document.getElementById("ImagenDistribuciones").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/cluster_distributions.png?" + timestamp;
    
    document.getElementById("tSNEClusters").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/tsne_clusters.png?" + timestamp;
  
    document.getElementById("descargarResultados").src = 
    "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/clusters.csv?" + timestamp;  
  }
  
  function actualizarImagenesGMM() {
    const timestamp = new Date().getTime(); // Para evitar caché, se agrega un timestamp único
    
    document.getElementById("ImagenDistrProbabilidadesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_probabilities.png?" + timestamp;
    
    document.getElementById("ImagenDistribucionesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/cluster_distributions.png?" + timestamp;
    
    document.getElementById("tSNEClustersGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/tsne_clusters.png?" + timestamp;
  
    document.getElementById("descargarResultadosGMM").src = 
    "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_clusters.csv?" + timestamp;  
  }
  