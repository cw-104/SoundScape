id = sessionStorage.getItem("uploadID");
if (id === null) {
  window.location.href = "index.html";
}

const resultCard = document.getElementById("result-card");
const explainationCard = document.getElementById("explaination-card");
const identifiedArtistCard = document.getElementById("identity-card");
const cardTitle = document.getElementById("result-title");
const cardBody = document.getElementById("result-body");
const advancedResultsBody = document.getElementById("advanced-results-body");
const explainationBody = document.getElementById("explaination-body");
const identifiedArtistBody = document.getElementById("identification-body");
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
  if (!data) {
    console.log("Error: Results not received");
    console.log(data);
    return;
  }
  displayResults(data);
}

function displayResults(results) {
  console.log(results);
  console.log(results.explaination);
  console.log(results.artist_id);

  explainationBody.innerHTML = results.explaination;
  explainationCard.classList.add("border-primary");
  identifiedArtistBody.innerHTML = results.artist_id;
  identifiedArtistCard.classList.add("border-primary");

  //   trim to 2 dec

  let finalPrediction = results.combined;
  let pretty_pred = Math.min(Math.round(finalPrediction.prediction * 10000) / 100); // 2 dec places as percentage not float
  let conf = "low"
  if (results.combined.prediction > .50) {
    conf = "High"
  } else if (results.combined.prediction > .30) {
    conf = "Medium"
  }

  let str =
    "Our system has " + conf + " confidence that this is ";
  if (finalPrediction.label === "Real") {
    resultCard.classList.add("border-success");
    cardTitle.innerHTML = "Real";
    cardBody.innerHTML = str + " not a deepfake.";
  } else {
    resultCard.classList.add("border-danger");
    cardTitle.innerHTML = "Fake";
    cardBody.innerHTML = str + "a deepfake.";
  }
  fillAdvancedResults(results['model_results']);
}

/**
 * @param {Map} results - results json
 */
function fillAdvancedResults(results) {
  /*
    html to insert into advanced-results-body
  */
  model_names = Object.keys(results)
  console.log(model_names)



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

  for (let model_name of model_names) {
    let model = results[model_name];
    separated = {
      "label": model["separated_results"]["label"],
      "prediction": model["separated_results"]["prediction"],
      "pretty_pred": Math.min(Math.round(model["separated_results"]["prediction"] * 10000) / 100), // 2 dec places as percentage not float
    }
    console.log(model.separated_results.prediction)
    if (model.separated_results.prediction > .50) {
      separated_cert = "High"
    } else if (model.separated_results.prediction > .15) {
      separated_cert = "Medium"
    } else {
      separated_cert = "Low"
    }

    html += `
       <tr class="table-dark">
            <th scope="row" class="text-body-secondary">${model_name}</th>
            <td class="text-body-secondary">${separated.label}</td>
            <td class="text-body-secondary">${separated_cert}</td>
        </tr>
        `;
  }
  html += `<tr class="table-active"><th class="text-info" colspan="3">Original</th></tr>`;

  for (let model_name of model_names) {
    let model = results[model_name];
    unseparated = {
      "label": model.unseparated_results.label,
      "prediction": model.unseparated_results.prediction,
      "pretty_pred": Math.min(Math.round(model.unseparated_results.prediction * 10000) / 100), // 2 dec places as percentage not float
    }


    if (model.unseparated_results.prediction > .50) {
      unseparated_cert = "High"
    } else if (model.unseparated_results.prediction > .15) {
      unseparated_cert = "Medium"
    } else {
      unseparated_cert = "Low"
    }

    html += `
        <tr class="table-dark">
            <th scope="row" class="text-body-secondary">${model_name}</th>
            <td class="text-body-secondary">${unseparated.label}</td>
            <td class="text-body-secondary">${unseparated_cert}</td>
        </tr>
            `;
  }

  html += "</table>";

  advancedResultsBody.innerHTML = html;
}
