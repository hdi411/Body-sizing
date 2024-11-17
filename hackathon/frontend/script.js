const video = document.getElementById('video');
const preview = document.getElementById('preview');
const canvas = document.getElementById('canvas');
const instructionText = document.getElementById('instructionText');
const initialButtons = document.getElementById('initialButtons');
const shutterButton = document.getElementById('shutterButton');
const actionButtons = document.getElementById('actionButtons');
const results = document.getElementById('results');
const predictedMeasurements = document.getElementById('predictedMeasurements');
const suggestedSize = document.getElementById('suggestedSize');

let photoStep = 'front';
let currentPhoto = null;

function startCamera() {
  try {
    const stream = navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = stream;
    video.style.display = 'block';
    preview.style.display = 'none';
    initialButtons.style.display = 'none';
    shutterButton.style.display = 'block';
  } catch (err) {
    alert('Unable to access camera: ' + err.message);
  }
}

function takePhoto() {
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  currentPhoto = canvas.toDataURL('image/png');

  preview.src = currentPhoto;
  preview.style.display = 'block';
  video.style.display = 'none';
  shutterButton.style.display = 'none';
  actionButtons.style.display = 'flex';
}

function uploadPhoto() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        currentPhoto = e.target.result;
        preview.src = currentPhoto;
        preview.style.display = 'block';
        video.style.display = 'none';
        initialButtons.style.display = 'none';
        actionButtons.style.display = 'flex';
      };
      reader.readAsDataURL(file);
    }
  };
  input.click();
}

function retakePhoto() {
  currentPhoto = null;
  preview.style.display = 'none';
  video.style.display = 'block';
  shutterButton.style.display = 'block';
  actionButtons.style.display = 'none';
}

function confirmPhoto() {
  if (!currentPhoto) {
    alert('Please take or upload a photo first');
    return;
  }

  if (photoStep === 'front') {
    sessionStorage.setItem('frontPhoto', currentPhoto);
    photoStep = 'side';
    instructionText.textContent = 'Take or upload a side photo (align your right shoulder)';
    retakePhoto();
  } else {
    const frontPhoto = sessionStorage.getItem('frontPhoto');
    const sidePhoto = currentPhoto;

    if (!frontPhoto || !sidePhoto) {
      alert('Both front and side photos are required.');
      return;
    }

    try {
      const response = fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          front_image: frontPhoto.split(',')[1], // Remove Base64 prefix
          side_image: sidePhoto.split(',')[1]
        })
      });

      const result = response.json();
      console.log('Server Response:', result); // Debugging: Inspect server response

      if (response.ok) {
        if (result.predicted_measurements && result.suggested_size) {
          // Display predicted measurements and suggested size
          predictedMeasurements.textContent = `Predicted Measurements:\n${JSON.stringify(result.predicted_measurements, null, 2)}`;
          suggestedSize.textContent = `Suggested Size: ${result.suggested_size}`;
          results.style.display = 'block';
        } else {
          console.error('Response format is invalid:', result);
          alert('Invalid response format from server.');
        }
      } else {
        console.error('Server error:', result);
        alert(`Error: ${result.error || 'Unknown server error.'}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert(`Failed to connect to the server: ${error.message}`);
    }

    sessionStorage.removeItem('frontPhoto');
    instructionText.textContent = 'Take or upload a front photo';
    photoStep = 'front';
    retakePhoto();
  }
}
