// Drag and Drop File Upload Functionality
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-select");
const fileHint = document.getElementById("file-hint");

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

function defFileBrowseEvents() {
  document
    .getElementById("file-browse-button")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      console.log("hello");
    });
}

// Prevent default behaviors
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

defFileDropEvents();
defFileBrowseEvents();

// Handle dropped files
function handleDrop(e) {
  files = e.dataTransfer.files;
  if (files.length > 1) {
    displayFileOrMessage("Please only upload one file at a time.");
    return;
  }
  file = files[0];

  if (file.type !== "audio/mpeg") {
    displayFileOrMessage("Please upload an mp3 file.");
    return;
  }
  console.log(file.name);
  fileSelect(file);
}

// Handle selected files
function fileSelect(file) {
  console.log(file);
  displayFileOrMessage(file.name);
}

function displayFileOrMessage(message) {
  fileHint.innerHTML = message;
}
