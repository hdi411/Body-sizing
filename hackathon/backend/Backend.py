import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added CORS
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import base64
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Example size chart
size_chart = {
    "XX-Small": {"chest": (78.7, 81.3), "waist": (64.8, 69.3)},
    "X-Small": {"chest": (83.8, 86.4), "waist": (69.8, 73.4)},
    "Small": {"chest": (88.9, 94.0), "waist": (74.9, 81.0)},
    "Medium": {"chest": (97.0, 102.9), "waist": (81.9, 89.9)},
    "Large": {"chest": (105.9, 111.8), "waist": (90.0, 95.0)},
}

# Step 1: Define the Model
class DualInputBodyMeasurementModel(torch.nn.Module):
    def __init__(self, num_measurements):
        super(DualInputBodyMeasurementModel, self).__init__()
        self.front_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.side_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.front_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.side_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        combined_features = self.front_model.fc.in_features + self.side_model.fc.in_features
        self.front_model.fc = torch.nn.Identity()
        self.side_model.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(combined_features, num_measurements)

    def forward(self, front_image, side_image):
        front_features = self.front_model(front_image)
        side_features = self.side_model(side_image)
        combined = torch.cat((front_features, side_features), dim=1)
        return self.fc(combined)

# Step 2: Load the Model
def load_model(model_path, num_measurements):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualInputBodyMeasurementModel(num_measurements=num_measurements).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
        return model, device
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise

# Step 3: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Step 4: Compare with Size Chart
def find_suitable_size(predicted, size_chart):
    for size, measurements in size_chart.items():
        waist_min, waist_max = measurements["waist"]
        chest_min, chest_max = measurements["chest"]

        if waist_min <= predicted["waist"] <= waist_max:
            if predicted["chest"] <= chest_max:
                return size
    return "No suitable size found"

# Step 5: Flask Application
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'front_image' not in data or 'side_image' not in data:
            return jsonify({'error': 'Both front and side images are required'}), 400

        # Decode base64 images
        front_image_data = base64.b64decode(data['front_image'])
        side_image_data = base64.b64decode(data['side_image'])

        # Load images as PIL Images
        front_image = Image.open(BytesIO(front_image_data)).convert("L")
        side_image = Image.open(BytesIO(side_image_data)).convert("L")
        logging.info(f"Front image size: {front_image.size}, Side image size: {side_image.size}")

        # Transform images for model input
        front_tensor = transform(front_image).unsqueeze(0).to(device)
        side_tensor = transform(side_image).unsqueeze(0).to(device)

        # Predict measurements
        with torch.no_grad():
            predictions = model(front_tensor, side_tensor).cpu().numpy().flatten()

        measurement_names = [
            "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
            "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"
        ]

        # Convert predicted values to standard Python floats
        predicted_measurements = {name: round(float(value), 2) for name, value in zip(measurement_names, predictions)}

        # Determine suitable size
        size = find_suitable_size(predicted_measurements, size_chart)

        # Format the response
        response = {
            "Predicted Measurements": predicted_measurements,
            "Suggested Size": size
        }

        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    try:
        model_path = "dual_body_measurement_model.pth"  # Path to your trained model
        num_measurements = 16  # Replace with your model's output count
        model, device = load_model(model_path, num_measurements)
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Failed to start the application: {e}", exc_info=True)
