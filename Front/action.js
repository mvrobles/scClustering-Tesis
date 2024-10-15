
async function LeerArchivos() {
  // Barcodes
  const fileInputBarcodes = document.getElementById('inputBarcodes');
  const fileBarodes = fileInputBarcodes.files[0];
  const formDataBarcodes = new FormData();
  formDataBarcodes.append('file_barcodes', fileBarodes);

  // Genes
  const fileInpuGenes = document.getElementById('inputGenes');
  const fileGenes = fileInpuGenes.files[0];
  const formDataGenes = new FormData();
  formDataGenes.append('file_genes', fileGenes);

  // Matrix
  const fileInpuMatrix = document.getElementById('inputMatrix');
  const fileMatrix = fileInpuMatrix.files[0];
  const formDataMatirix = new FormData();
  formDataMatirix.append('file_mtx', fileMatrix);

  // Create a new FormData object to combine the three
  const combinedFormData = new FormData();
  function appendFormData(target, source) {
    for (let pair of source.entries()) {
      target.append(pair[0], pair[1]);
    }
  }

  // Append the contents of each FormData object to the combined FormData object
  appendFormData(combinedFormData, formDataBarcodes);
  appendFormData(combinedFormData, formDataGenes);
  appendFormData(combinedFormData, formDataMatirix);

  const response = await fetch('http://127.0.0.1:8080/lecturaArchivos/', {
    method: "POST",
    body: combinedFormData
  })

  const responseText = await response.text();
  console.log(responseText); // logs 'OK'


  console.log(JSON.parse(responseText));
  if (JSON.parse(responseText).status === 200) {
    var index_page = document.getElementById("mensajeCarga"); 
    index_page.style.color = "black"; 
    const responseData = JSON.parse(responseText);
    index_page.innerHTML = `¡Carga exitosa! Se procesó el experimento de scRNA-seq con ${responseData.num_celulas} células y ${responseData.num_genes} genes.`;

    document.getElementById("graph-based").style.visibility = "visible";
    document.getElementById("neural-network").style.visibility = "visible";
    document.getElementById("bayesian-models").style.visibility = "visible";
    document.getElementById("seleccione-algoritmo-text").style.visibility = "visible";
  }
}

document.addEventListener("DOMContentLoaded", function() {
  const button1 = document.getElementById("graph-based");
  if (button1) {
    document.getElementById("graph-based").addEventListener("click", function() {
      window.location.href = "graphBasedAlgorithm.html";
    });
  }

  const button2 = document.getElementById("neural-network");
  if (button2) {
    document.getElementById("neural-network").addEventListener("click", function() {
      window.location.href = "NNBasedAlgorithm.html";
    });
  }

  const button3 = document.getElementById("bayesian-models");
  if (button3){
    document.getElementById("bayesian-models").addEventListener("click", function() {
      window.location.href = "NaiveBayesAlgorithm.html";
    });
  }
});

async function CorrerModeloGrafos(){
  const response = await fetch('http://127.0.0.1:8080/CorrerModeloGrafos/', {
    method: "POST",
  });

  console.log(response.status);
  const responseText = await response.text();
  if (response.status === 200) {
    console.log("ENTRO AL IF");
    actualizarImagenesGraph();
    document.getElementById("ImagenCorrelaciones").style.visibility = "visible";
    document.getElementById("ImagenDistribuciones").style.visibility = "visible";
    document.getElementById("tSNEClusters").style.visibility = "visible";
    document.getElementById("descargarResultados").style.visibility = "visible";
  }
}

async function CorrerModeloGMM(){
  let n_clusters = document.getElementById("n_clusters").value;
  console.log("Número de clusters:");
  console.log(n_clusters);
  const response = await fetch('http://127.0.0.1:8080/CorrerModeloGMM/', {
    method: "POST",
    body: JSON.stringify({n_clusters:n_clusters})
  });

  console.log("Terminó de correr el modelo GMM");

  console.log(response.status);
  if (response.status === 200) {
    actualizarImagenesGMM()
    document.getElementById("ImagenDistrProbabilidadesGMM").style.visibility = "visible";
    document.getElementById("ImagenDistribucionesGMM").style.visibility = "visible";
    document.getElementById("tSNEClustersGMM").style.visibility = "visible";
    document.getElementById("descargarResultadosGMM").style.visibility = "visible";
  }
}

async function CorrerModeloNB(){
  const tejidoSeleccionado = document.querySelector('.dropdown-toggle').textContent;
  console.log("Tejido:");
  console.log(n_clusters);
  const response = await fetch('http://127.0.0.1:8080/CorrerModeloNB/', {
    method: "POST",
    body: JSON.stringify({n_clusters:n_clusters})
  });

  console.log(response.status);
  if (response.status === 200) {
    actualizarImagenes()
    document.getElementById("ImagenCorrelaciones").style.visibility = "visible";
    document.getElementById("ImagenDistribuciones").style.visibility = "visible";
    document.getElementById("tSNEClusters").style.visibility = "visible";
  }
}

async function SubmitVars() {
    console.log("Entro");
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    console.log(formData);
    const response = await fetch('http://127.0.0.1:8080/lecturaDatos/', {
    method: "POST",
    body: formData
  })

  const responseText = await response.text();
  console.log(responseText); // logs 'OK'
  var index_page = document.getElementById("answer"); 
  index_page.style.color = "blue"; 
  if(JSON.stringify(responseText).indexOf('overlap') > -1){index_page.style.color = "red"};
  index_page.innerHTML = (responseText);
}

async function ReadJavaData() {

  const fileInput = document.getElementById('fileInputJava');
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8888/readCsvFile', {
    method: "POST",
    body: formData
  })

  const responseText = await response.text();
  console.log(responseText); // logs 'OK'
  var index_page = document.getElementById("answerJava2"); 
  index_page.style.color = "blue"; 
  if(JSON.stringify(responseText).indexOf('overlap') > -1){index_page.style.color = "red"};
  index_page.innerHTML = (responseText);
}