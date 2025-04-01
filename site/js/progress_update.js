let progressLabel = null;
let seeResultsButton = null;
let id = null;
substate = null;
function redirectIfNoID() {
  let id = sessionStorage.getItem("uploadID");
  if (id === null) {
    window.location.href = "index.html";
  }
}
function initProgress() {
  id = sessionStorage.getItem("uploadID");

  /** @type {Array<[number, number, string]>} */
  const steps = [
    [0, 10, "bg-info"],
    [0, 80, "bg-success"],
    [0, 10, "bg-warning"],
  ];
  addProgressBars(steps);

  progressLabel = document.getElementById("progress-state");
  seeResultsButton = document.getElementById("results-button");

  displayProgressHint("Uploading...");
  update();
}

function displayProgressHint(str) {
  progressLabel.innerHTML = str;
  if (substate) {
    progressLabel.innerHTML += ` (${substate})`;
  }
}

function getProgressHint() {
  return progressLabel.innerHTML;
}

let prevState = -1;
let times_checked = 0;
let prev_partial = 0;
async function update() {
  console.log(`Fetching state for ID: ${id}`);
  await fetchStatus(id).then(async (data) => {
    if (!data) {
      console.log("Error: data not received");
      return;
    }

    if (data.substate) {
      console.log("Substate: " + data.substate);
      substate = data.substate
    } else substate = null;

    let state = parseState(data);
    if (state != prevState) {
      times_checked = 1;
      prev_partial = 1;
    }
    prevState = state;
    function progression_decay(n) {
      let max = 140;
      let speed = 20; // lower means faster start
      return Math.min(max - (max * speed) / (speed + n), 80);
    }
    let partial = progression_decay(times_checked);
    let update_wait = 3000;
    const start = performance.now();
    let fast_incr = 15;
    let fast_wait = 100;

    let partial_incr = 1;
    let partial_wait = ((partial - prev_partial) / 100) * update_wait;
    partial_wait = Math.max(10, Math.round(partial_wait));
    prev_partial = partial;

    // update status
    if (state === STATES.NONE) {
      console.log("Error: No state received");
      displayProgressHint("Error try again");
    } else if (state === STATES.UPLOADED) {
      await animate_progress(0, 100, fast_incr, fast_wait);
      displayProgressHint("Analyzing...");
      await animate_progress(1, partial, partial_incr, partial_wait);
    } else if (state === STATES.PROCESSING) {
      await animate_progress(0, 100, fast_incr, fast_wait);
      displayProgressHint("Analyzing...");
      await animate_progress(1, partial, partial_incr, partial_wait);
    } else if (state === STATES.FINISHED) {
      await animate_progress(0, 100, fast_incr, fast_wait);
      displayProgressHint("Analyzing...");
      await animate_progress(1, 100, fast_incr, fast_wait);
      displayProgressHint("Gathering Results...");
      await animate_progress(2, 100, fast_incr, fast_wait);
      displayProgressHint("Finished!");
      prepareResults();
      return;
    } else {
      console.log("not updating progress");
      displayProgressHint("Error try again");
    }



    new Promise((r) =>
      setTimeout(r, Math.max(10, update_wait - (performance.now() - start)))
    ).then(() => {
      times_checked++;
      update();
    });
  });
}

async function prepareResults() {
  // enable button
  seeResultsButton.innerHTML = "See Results";
  seeResultsButton.disabled = false;
  seeResultsButton.addEventListener("click", () => {
    window.location.href = "results.html";
  });
}

function updateProgressLabel(status) {
  if (status === 0) {
    progressLabel.innerHTML = "Uploading...";
  } else if (status === 1) {
    progressLabel.innerHTML = "Processing...";
  } else if (status === 2) {
    progressLabel.innerHTML = "Finished!";
  } else {
    progressLabel.innerHTML = "Error";
  }
}
