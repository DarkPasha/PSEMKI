const image_input = document.querySelector("#image-input");
var uplouad_image = "";

image_input.addEventListener("change", function() {
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    uploaded_image = reader.result;
    document.querySelector("#display-image").style.backgroundImage = `url(${uploaded_image})`;
  });
  reader.readAsDataURL(this.files[0]);
});

function PythonAusführen(){
  $.ajax({
    url: "catsanddogs_final_laptopvers.py",
    context: document.body});
  }
