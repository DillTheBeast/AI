// Create the main container
var body = document.body;

// Create the header
var header = document.createElement("header");
header.style.backgroundColor = "#333";
header.style.color = "white";
header.style.textAlign = "center";
header.style.padding = "1em";
header.innerHTML = "<h1>Welcome to My Website</h1>";
body.appendChild(header);

// Create the main content section
var main = document.createElement("main");
main.style.padding = "20px";
body.appendChild(main);

// Create a section with content and a button
var contentSection = document.createElement("section");
contentSection.id = "content";
contentSection.style.maxWidth = "600px";
contentSection.style.margin = "0 auto";
contentSection.innerHTML = "<p>This is a simple JavaScript website.</p>";

var button = document.createElement("button");
button.style.backgroundColor = "#4CAF50";
button.style.color = "white";
button.style.padding = "10px 20px";
button.style.fontSize = "16px";
button.style.cursor = "pointer";
button.innerHTML = "Click me";

button.addEventListener("click", function () {
    contentSection.innerHTML = "Button clicked! Text changed by JavaScript.";
});

contentSection.appendChild(button);
main.appendChild(contentSection);
