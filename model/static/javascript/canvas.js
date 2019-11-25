// Adapted from https://stackoverflow.com/a/8398189/8721358

var canvas, ctx, flag = false,
  prevX = 0,
  currX = 0,
  prevY = 0,
  currY = 0,
  dot_flag = false;

/*
 * Sets the color and size of the brush
 * From testing, setting the color of the brush to black appears to cause inaccuracies in the prediction of
 * the digit. This is due to the use of cv2
 */
var x = "red",
  y = 3;

// Sets the url that the user drawn image will be posted to 
var url = "http://127.0.0.1:5000/digit";

// Initialise the canvas and set it up
function init() {
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext("2d");
  w = canvas.width;
  h = canvas.height;

  canvas.addEventListener("mousemove", function (e) {
    findxy('move', e)
  }, false);
  canvas.addEventListener("mousedown", function (e) {
    findxy('down', e)
  }, false);
  canvas.addEventListener("mouseup", function (e) {
    findxy('up', e)
  }, false);
  canvas.addEventListener("mouseout", function (e) {
    findxy('out', e)
  }, false);
}

// Allows the user to draw with their mouse
function draw() {
  ctx.beginPath();

  ctx.moveTo(prevX, prevY);
  ctx.lineTo(currX, currY);

  ctx.strokeStyle = x;
  ctx.lineWidth = y;

  ctx.stroke();
  ctx.closePath();
}

// Clears the canvas 
function erase() {
  ctx.clearRect(0, 0, w, h);
}

// Submits the image to the model
function submitImage() {
  var imageData = canvas.toDataURL();
  
  $.post(url, {
    "imageBase64": imageData
  }, function(data) {
    // document.getElementById("predictedNumber").innerHTML = data;
    // $("#predictedNumberHeader").append(data);
    $("#predictedNumber").empty().append(data);
  });
}


function findxy(res, e) {
  if (res == 'down') {
    prevX = currX;
    prevY = currY;
    currX = e.clientX - canvas.offsetLeft;
    currY = e.clientY - canvas.offsetTop;

    flag = true;
    dot_flag = true;

    if (dot_flag) {
      ctx.beginPath();

      ctx.fillStyle = x;
      ctx.fillRect(currX, currY, 2, 2);

      ctx.closePath();
      dot_flag = false;
    }
  }

  if (res == 'up' || res == "out") {
    flag = false;
  }

  if (res == 'move') {
    if (flag) {
      prevX = currX;
      prevY = currY;

      currX = e.clientX - canvas.offsetLeft;
      currY = e.clientY - canvas.offsetTop;

      draw();
    }
  }
}