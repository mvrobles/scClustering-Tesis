function actualizarImagenesGraph() {
    const timestamp = new Date().getTime();
    
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
    const timestamp = new Date().getTime();
    
    document.getElementById("ImagenDistrProbabilidadesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_probabilities.png?" + timestamp;
    
    document.getElementById("ImagenDistribucionesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/cluster_distributions.png?" + timestamp;
    
    document.getElementById("tSNEClustersGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/tsne_clusters.png?" + timestamp;
  
    document.getElementById("descargarResultadosGMM").src = 
    "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_clusters.csv?" + timestamp;  
  }

  function actualizarImagenesNN() {
    const timestamp = new Date().getTime();
    
    document.getElementById("ImagenDistrProbabilidadesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_probabilities.png?" + timestamp;
    
    document.getElementById("ImagenDistribucionesGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/cluster_distributions.png?" + timestamp;
    
    document.getElementById("tSNEClustersGMM").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/tsne_clusters.png?" + timestamp;
  
    document.getElementById("descargarResultadosGMM").src = 
    "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/gmm_clusters.csv?" + timestamp;  
  }

  function actualizarImagenesNB() {
    const timestamp = new Date().getTime();
    
    document.getElementById("ImagenTissues").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/nb_tissue_distribution.png?" + timestamp;
    
    document.getElementById("ImagenTissuesTipos").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/nb_type_distribution.png?" + timestamp;
    
    document.getElementById("tSNEClustersNB").src = 
        "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/tsne_clusters.png?" + timestamp;
  
    document.getElementById("descargarResultadosNB").src = 
    "/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp/results/nb_clusters.csv?" + timestamp;  
  }
  