// Fonction pour accéder à la caméra
async function startCamera() {
  try {
    // Demander l'accès à la caméra
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    
    // Récupérer l'élément vidéo et lui affecter le flux
    const videoElement = document.getElementById('video');
    videoElement.srcObject = stream;
  } catch (error) {
    console.error("Erreur lors de l'accès à la caméra : ", error);
    alert("Impossible d'accéder à la caméra !");
  }
}

// Démarrer la caméra lorsque la page est chargée
window.onload = startCamera;
