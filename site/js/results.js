id = sessionStorage.getItem("uploadID");
if (id === null) {
  window.location.href = "index.html";
}

const resultCard = document.getElementById("result-card");
const cardTitle = document.getElementById("result-title");
const cardBody = document.getElementById("result-body");
const advancedResultsBody = document.getElementById("advanced-results-body");
prep();

async function prep() {
  let data = await fetchStatus(id);
  let state = parseState(data);
  if (state != STATES.FINISHED) {
    window.location.href = "progress.html";
  }

  data = await fetchResults(id);
  console.log(data);
  data = JSON.parse(data);
  if (!data || !data.status) {
    console.log("Error: Results not received");
    console.log(data);
    return;
  }
  if (data.status !== "finished") {
    console.log("Error: Results returned finished even after status check");
  }

  displayResults(data);
}

function displayResults(json) {
  let resultSet = new ModelResultsSet(json);
  console.log(resultSet);

  let success = resultSet.rawgat.separated.label === "Real";
  let confidence = resultSet.rawgat.separated.pred;
  //   trim to 2 dec
  confidence = Math.round(confidence * 10000) / 100;

  let str = "Our system is " + confidence + "% confident that this is ";
  if (success) {
    resultCard.classList.add("border-success");
    cardTitle.innerHTML = "Real";
    cardBody.innerHTML = str + " not a deepfake.";
  } else {
    resultCard.classList.add("border-danger");
    cardTitle.innerHTML = "Fake";
    cardBody.innerHTML = str + "a deepfake.";
  }
  fillAdvancedResults(resultSet);
}

/**
 * @param {ModelResultsSet} resultSet - The name of the user.
 */
function fillAdvancedResults(resultSet) {
  /*
        html to insert into advanced-results-body  
    */

  /**
   * @type {ModelResults[]} models
   */
  let models = [resultSet.whisper, resultSet.rawgat];

  let html = `
<table class="table table-hover">
  <thead>
      <tr>
      <th scope="col">Model</th>
      <th scope="col">Label</th>
      <th scope="col">Certainty</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
    `;

  html += `<tr class="table-active"><th class="text-info" colspan="3">Vocals Only</th></tr>`;
  for (let model of models) {
    html += `
       <tr class="table-dark">
            <th scope="row" class="text-body-secondary">${model.name}</th>
            <td class="text-body-secondary">${model.separated.label}</td>
            <td class="text-body-secondary">${model.separated.pred}%</td>
        </tr>
        `;
  }
  html += `<tr class="table-active"><th class="text-info" colspan="3">Original</th></tr>`;
  for (let model of models) {
    html += `
        <tr class="table-dark">
            <th scope="row" class="text-body-secondary">${model.name}</th>
            <td class="text-body-secondary">${model.unseparated.label}</td>
            <td class="text-body-secondary">${model.unseparated.pred}%</td>
        </tr>
            `;
  }

  html += "</table>";

  advancedResultsBody.innerHTML = html;
}
