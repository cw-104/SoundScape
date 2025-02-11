// Drag and Drop File Upload Functionality
const dropArea = document.getElementById("drop-area");
const fileHint = document.getElementById("file-hint");
const fileInput = document.getElementById("file-select");
const submitButton = document.getElementById("analyze-button");

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
      displayFileOrMessage("Upload failed. Try again.");
    }
  });
});
function insertProgress() {
  displayFileOrMessage("File uploaded!");
  // disable submit button
  // @ts-ignore
  submitButton.disabled = true;
  submitButton.innerHTML = "Analyzing...";
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
    // @ts-ignore
    fileInput.value = "";
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
  if (files.length > 1) {
    displayFileOrMessage("Please only upload one file at a time.");
    return false;
  } else if (files.length === 0) {
    displayFileOrMessage("Please upload an mp3 file.");
    return false;
  }
  let file = files[0];

  if (file.type !== "audio/mpeg") {
    displayFileOrMessage("Uploads must be in mp3 format.");
    return false;
  }
  displayFileOrMessage("File ready to analyze!");
  return true;
}

// Handle dropped files
function handleDrop(e) {
  let files = e.dataTransfer.files;
  if (!checkFiles(files)) return;

  // @ts-ignore
  fileInput.files = files; // pass to the file input form
}

function displayFileOrMessage(message) {
  fileHint.innerHTML = message;
}

/**
 *
 * @returns {Promise<string>}
 */
async function uploadToAPI(file) {
  let data = new FormData();
  data.append("audio", file);
  displayFileOrMessage("Uploading...");

  return await fetch(getURL() + "/upload", { method: "POST", body: data })
    .then((response) => response.json())
    .then((body) => {
      if (body.id) {
        return body.id;
      } else {
        console.error("body.id not found");
        displayFileOrMessage("Upload failed. Try again.");
        return;
      }
    })
    .catch((error) => {
      console.error("Upload Error:", error);
      displayFileOrMessage("Upload failed. Try again.");
      return;
    });
}
