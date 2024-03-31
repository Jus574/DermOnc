document.addEventListener("DOMContentLoaded", function () {
  let slides = document.querySelectorAll(".slide");
  let currentSlide = 0;

  function showSlide(n) {
    slides[currentSlide].classList.remove("active");
    currentSlide = (n + slides.length) % slides.length;
    slides[currentSlide].classList.add("active");
  }

  function nextSlide() {
    showSlide(currentSlide + 1);
  }

  setInterval(nextSlide, 2000); // Change slide every 2 seconds
});

document.addEventListener("DOMContentLoaded", function () {
  var dropdownList = document.getElementById("dropdown-list");
  dropdownList.style.display = "none";
});

function toggleDropdown() {
  var dropdownList = document.getElementById("dropdown-list");
  if (
    dropdownList.style.display === "none" ||
    dropdownList.style.display === ""
  ) {
    dropdownList.style.display = "block";
  } else {
    dropdownList.style.display = "none";
  }
}

function scrollToSection(sectionId) {
  // Scroll to the section with the specified ID
  document.getElementById(sectionId).scrollIntoView({ behavior: "smooth" });
}

function redirectToMap() {
  window.location.href = "find-specialists";
}

document.getElementById("file").addEventListener("change", function () {
  var file = this.files[0];
  if (file) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var imageSrc = e.target.result;
      document.getElementById("uploadedImage").src = imageSrc;
      document.getElementById("popup").style.display = "block";
      document.getElementById("Loading-Btn").style.display = "block";
    };
    reader.readAsDataURL(file);
  }
  document
    .getElementById("submitBtn")
    .addEventListener("click", function (event) {
      document.getElementsByClassName("submitBtn")[0].className +=
        " is-loading";

      setTimeout(function () {
        document.getElementsByClassName("submitBtn")[0].className = "loading";
      }, 4000);

      setTimeout(function () {
        loading.innerHTML = "Submit";
      }, 4100);

      document.getElementById("uploadForm").submit();
    });
});

function redirectToHome() {
  window.location.href = "index";
}

function redirectToInfo() {
  window.location.href = "info";
}

function redirectToMelanoma() {
  window.location.href = "melanoma";
}
