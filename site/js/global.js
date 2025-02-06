// Example of changing the URL without reloading
function navigateTo(page) {
  history.pushState(null, "", page);
  // Load the content dynamically (e.g., using fetch)
}

// Fetch navbar
fetch("navbar.html") // Assuming your navbar code is in navbar.html
  .then((response) => response.text())
  .then((data) => {
    document.getElementById("navbar-container").innerHTML = data;
  });
