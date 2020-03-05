let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rakib=0, nazmul=0, shemul=0,mayen=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			rakib++;
			document.getElementById("rakibsamples").innerText = "Rakib's samples:" + rakib;
			break;
		case "1":
			nazmul++;
			document.getElementById("nazmulsamples").innerText = "Nazmul's samples:" + nazmul;
			break;
		case "2":
			shemul++;
			document.getElementById("shemulsamples").innerText = "Shemul's samples:" + shemul;
			break;
		case "3":
			mayen++;
			document.getElementById("mayensamples").innerText = "Mayen's samples:" + mayen;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    var d = new Date();
    switch(classId){
		case 0:
			predictionText = "Rakib last seen at : ";
			break;
		case 1:
			predictionText = "Nazmul last seen at : ";
			break;
		case 2:
			predictionText = "Shemul last seen at : ";
			break;
		case 3:
			predictionText = "Mayen last seen at : ";
			break;
	}
	document.getElementById("prediction").innerText = predictionText + d;
	
	//document.getElementById("demo").innerHTML = d.getDate();
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
