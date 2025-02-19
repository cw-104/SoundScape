const StatusEnum = Object.freeze({
  DISABLED: 0,
  READY: 1,
  UPLOADING: 2,
  IN_PROGRESS: 3,
});
const file_hint_initial_color = "text-warning";

// Drag and Drop File Upload Functionality
const dropArea = document.getElementById("drop-area");
const fileHint = document.getElementById("file-hint");
const fileInput = document.getElementById("file-select");
const submitButton = document.getElementById("analyze-button");
setButtonState(StatusEnum.DISABLED);
fileHint.classList.add(file_hint_initial_color);

submitButton.addEventListener("click", function (event) {
  event.preventDefault();
  console.log("Submit Button Clicked");
  // @ts-ignore
  if (!checkFiles(fileInput.files)) return;

  // @ts-ignore
  const file = fileInput.files[0];
  uploadToAPI(file).then((id) => {
    console.log("Upload ID:", id);
    if (id) {
      sessionStorage.setItem("uploadID", id);
      // window.location.href = "progress.html";
      insertProgress();
    } else {
      displayFileOrMessage("Upload failed. Try again.", true);
    }
  });
});

function setButtonState(status) {
  console.log(status);
  submitButton.disabled = status !== StatusEnum.READY;
  success_class = "btn-success";
  disabled_class = "btn-dark";
  uploading_class = "btn-outline-warning";
  in_progress_class = "btn-outline-success";

  switch (status) {
    case StatusEnum.DISABLED:
      submitButton.innerHTML = "Waiting for File";
      submitButton.classList.remove(success_class);
      submitButton.classList.remove(in_progress_class);
      submitButton.classList.remove(uploading_class);
      submitButton.classList.add(disabled_class);
      break;
    case StatusEnum.READY:
      submitButton.innerHTML = "Upload";
      submitButton.classList.remove(disabled_class);
      submitButton.classList.remove(in_progress_class);
      submitButton.classList.remove(uploading_class);
      submitButton.classList.add(success_class);
      break;
    case StatusEnum.UPLOADING:
      submitButton.innerHTML = "Uploading...";
      submitButton.classList.remove(disabled_class);
      submitButton.classList.remove(success_class);
      submitButton.classList.remove(in_progress_class);
      submitButton.classList.add(uploading_class);
      break;
    case StatusEnum.IN_PROGRESS:
      submitButton.innerHTML = "Analyzing...";
      submitButton.classList.remove(disabled_class);
      submitButton.classList.remove(success_class);
      submitButton.classList.remove(uploading_class);
      submitButton.classList.add(in_progress_class);
      break;
  }
}

function insertProgress() {
  displayFileOrMessage("File uploaded!", false);
  // disable submit button
  // @ts-ignore
  setButtonState(StatusEnum.IN_PROGRESS);
  fetch("html-elements/progress_bar.html")
    .then((response) => response.text())
    .then((data) => {
      const container = document.getElementById("insert-progress-container");
      container.style.display = "block";
      container.innerHTML = data;

      document.getElementById("progress-container-header").style.display =
        "block";
      initProgress();
    });
}

fileInput.addEventListener("change", function (event) {
  event.preventDefault();

  // @ts-ignore
  if (!checkFiles(event.target.files)) {
    return;
  }

  // @ts-ignore
  const file = fileInput.files[0];

  if (file) {
    console.log("File Name:", file.name);
    console.log("File Size:", file.size);
    console.log("File Type:", file.type);
  } else {
    console.log("No file selected.");
  }
});

function defFileDropEvents() {
  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // in drop zone
  ["dragenter", "dragover"].forEach((eventName) => {
    // highlight
    dropArea.addEventListener(
      eventName,
      function () {
        dropArea.classList.add("hover");
      },
      false
    );
  });

  // not in drop zone or dropped unhighlight
  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      function () {
        dropArea.classList.remove("hover");
      },
      false
    );
  });

  // on drop actual file handling
  dropArea.addEventListener("drop", handleDrop, false);
}

// Prevent default behaviors
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

defFileDropEvents();

function checkFiles(files) {
  setButtonState(StatusEnum.DISABLED);

  if (files.length > 1) {
    displayFileOrMessage("Please only upload one file at a time.", true);
    fileInput.value = "";
    return false;
  } else if (files.length === 0) {
    displayFileOrMessage("Please upload an mp3 file.", true);
    fileInput.value = "";
    return false;
  }
  let file = files[0];

  if (file.type !== "audio/mpeg") {
    displayFileOrMessage("Uploads must be in mp3 format.", true);
    fileInput.value = "";
    return false;
  }
  displayFileOrMessage("File ready to analyze!", false);
  setButtonState(StatusEnum.READY);
  return true;
}

// Handle dropped files
function handleDrop(e) {
  let files = e.dataTransfer.files;
  if (!checkFiles(files)) return;

  // @ts-ignore
  fileInput.files = files; // pass to the file input form
}

function displayFileOrMessage(message, is_error) {
  fileHint.classList.remove(file_hint_initial_color);
  if (!is_error) {
    fileHint.classList.remove("text-danger");
    fileHint.classList.add("text-success");
  } else {
    fileHint.classList.add("text-danger");
    fileHint.classList.remove("text-success");
  }
  fileHint.innerHTML = message;
}

/**
 *
 * @returns {Promise<string>}
 */
async function uploadToAPI(file) {
  let data = new FormData();
  data.append("audio", file);
  displayFileOrMessage("Uploading...", false);
  setButtonState(StatusEnum.UPLOADING);

  return await fetch(getURL() + "/upload", { method: "POST", body: data })
    .then((response) => response.json())
    .then((body) => {
      if (body.id) {
        return body.id;
      } else {
        console.error("body.id not found");
        displayFileOrMessage("Upload failed. Try again.", true);
        return;
      }
    })
    .catch((error) => {
      console.error("Upload Error:", error);
      displayFileOrMessage("Upload failed. Try again.", true);
      return;
    });
}
