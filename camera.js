// Fonction pour accéder à la caméra
async function startCamera() {
  try {
    // Demander l'accès à la caméra
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    
    // Récupérer l'élément vidéo et lui affecter le flux
    const videoElement = document.getElementById('video');
    videoElement.srcObject = stream;
    
    // Démarrer la détection d'émotions dès que la vidéo commence
    videoElement.onplay = () => detectEmotion(videoElement);
  } catch (error) {
    console.error("Erreur lors de l'accès à la caméra : ", error);
    alert("Impossible d'accéder à la caméra !");
  }
}

// Charger le modèle TensorFlow.js
async function loadModel() {
  const model = await tf.loadLayersModel('/home/cedric/Bureau/Travail/Arts Plastique/mon_modele_final.json');  // Remplace par le chemin vers ton modèle
  return model;
}

// Détecter les visages dans l'image
function detectFace(videoElement) {
  let mat = cv.imread(videoElement);
  let gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  
  // Charger la cascade de détection de visage
  let faceCascade = new cv.CascadeClassifier();
  faceCascade.load('haarcascade_frontalface_default.xml');
  
  let faces = new cv.RectVector();
  faceCascade.detectMultiScale(gray, faces);
  
  // Retourner le premier visage détecté
  let face = faces.size() > 0 ? faces.get(0) : null;
  
  mat.delete();
  gray.delete();
  faceCascade.delete();
  
  return face;
}

// Prédire l'émotion du visage détecté
async function predictEmotion(videoElement, model) {
  let face = detectFace(videoElement);
  
  if (face) {
    // Créer un contexte pour dessiner le visage détecté
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    
    // Dessiner le visage détecté
    ctx.drawImage(videoElement, face.x, face.y, face.width, face.height, 0, 0, canvas.width, canvas.height);
    
    // Convertir la zone du visage en TensorFlow.js tensor
    const faceRegion = tf.browser.fromPixels(canvas).resizeNearestNeighbor([128, 128]).toFloat().expandDims();
    
    // Prédire l'émotion
    const prediction = model.predict(faceRegion);
    const emotionIndex = prediction.argMax(-1).dataSync()[0];
    
    // Liste des émotions
    const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
    
    // Afficher l'émotion prédite
    const emotionText = emotions[emotionIndex];
    document.getElementById('emotion').innerText = `Émotion détectée : ${emotionText}`;
  }
}

// Fonction principale pour gérer la caméra et la détection d'émotions
async function detectEmotion(videoElement) {
  const model = await loadModel();  // Charger le modèle une fois
  setInterval(() => {
    predictEmotion(videoElement, model);  // Appliquer la détection sur chaque frame
  }, 100);  // Intervalle de 100 ms pour analyser la vidéo
}

// Démarrer la caméra au chargement de la page
