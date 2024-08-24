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
}

async function CorrerModelo(){
  const response = await fetch('http://127.0.0.1:8080/CorrerModelo/', {
    method: "POST",
  })
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


async function GreetUser() {
    console.log(JSON.stringify({"name": "Melissa", "age": 26}))
    const response = await fetch('http://127.0.0.1:8888/api/greet', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
        },
      body: JSON.stringify({"name": "Melissa", "age": 26})
    })
  
    const responseText = await response.text();
    console.log(responseText); // logs 'OK'
    var index_page = document.getElementById("answerJava"); 
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