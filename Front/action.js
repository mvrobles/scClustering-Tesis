async function SubmitVars() {

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

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