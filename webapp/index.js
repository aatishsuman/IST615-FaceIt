let model;
const webcam = new Webcam(document.getElementById('wc'));
let isPredicting = false;

async function init(){
    document.getElementById("message").innerText = "Loading model ...";
    await webcam.setup();
	model = await tf.loadLayersModel('http://127.0.0.1:8887/model.json');
	tf.tidy(() => model.predict(webcam.capture()));
    document.getElementById("message").innerText = "Model loaded";
}

init();

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
        console.log(model)
      const predictions = model.predict(img);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    predictionText = (classId == 1) ? "with mask" : "without mask"
	document.getElementById("prediction").innerText = predictionText;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}
