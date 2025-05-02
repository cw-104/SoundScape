let url = "http://127.0.0.1:8080"

// Fetch navbar
fetch("html-elements/navbar.html")
  .then((response) => response.text())
  .then((data) => {
    document.body.insertAdjacentHTML("afterbegin", data);
    const path = window.location.pathname;

    // ----- upload -----
    const uploadLink = document.getElementById("upload-link");
    // @ts-ignore
    uploadLink.href = "upload";

    const homeLink = document.getElementById("home-link");

    // ----- home -----
    // @ts-ignore
    homeLink.href = "home";
    // @ts-ignore
    document.getElementById("nav-brand").href = "home";

    // ----- about -----
    const aboutLink = document.getElementById("about-link");
    // @ts-ignore
    aboutLink.href = "about";

    // ----- credits -----
    const creditsLink = document.getElementById("credits-link");
    // @ts-ignore
    creditsLink.href = "credits";

    if (path.includes("home")) {
      homeLink.classList.add("active");
      // @ts-ignore
      document.getElementById("nav-brand").href = "#";
      // @ts-ignore
      homeLink.href = "#";
    } else if (path.includes("about")) {
      aboutLink.classList.add("active");
      // @ts-ignore
      aboutLink.href = "#";
    } else if (path.includes("credits")) {
      creditsLink.classList.add("active");
      // @ts-ignore
      creditsLink.href = "#";
    } else if (path.includes("upload")) {
      uploadLink.classList.add("active");
      // @ts-ignore
      uploadLink.href = "#";
    }
  });

// let url = "http://127.0.0.1:8080";

function getURL() {
  return url;
}

const STATES = {
  NONE: 0,
  UPLOADED: 1,
  PROCESSING: 2,
  FINISHED: 3,
  ERROR: -1,
};

async function fetchStatus(id) {
  let response = await fetch(getURL() + "/status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: id }),
  });
  if (!response.ok) {
    console.log("Error");
    return;
  }
  let data = await response.json();
  return data;
}

function parseState(data) {
  if (!data.state) {
    console.log("Data: " + data);
    return STATES.NONE;
  }
  let state = data.state.toLowerCase();
  if (state === "uploaded") {
    return STATES.UPLOADED;
  } else if (state === "processing") {
    return STATES.PROCESSING;
  } else if (state === "finished") {
    return STATES.FINISHED;
  } else {
    return STATES.ERROR;
  }
}
async function fetchResults(id) {
  console.log(`Fetching results for ID: ${id}`);

  let response = await fetch(getURL() + "/resultsv2", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: id }),
  });
  if (!response.ok) {
    console.log("Error");
    return;
  }
  let data = await response.json();
  return data;
}

class ModelResultsSet {
  constructor(json) {
    this.whisper = new ModelResults(json.whisper, "Whisper");
    this.rawgat = new ModelResults(json.rawgat, "Rawgat");
    this.xlsr = new ModelResults(json.xlsr, "xlsr");
    this.vocoder = new ModelResults(json.vocoder, "Vocoder");
  }
}
class ModelResults {
  constructor(model_json, str_name) {
    this.name = str_name;
    this.separated = new Prediction(model_json.unseparated_results);
    this.unseparated = new Prediction(model_json.separated_results);
  }
}
class Prediction {
  constructor(res_json) {
    this.label = res_json.label;
    this.pred = res_json.prediction;
    this.pretty_pred = Math.min(Math.round(this.pred * 10000) / 100); // 2 dec places as percentage not float
  }
}
