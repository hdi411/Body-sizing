import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd


# Step 1: Load and Merge Data
def load_and_merge_data(subject_to_photo_path, measurements_path, hwg_metadata_path):
    subject_to_photo_map = pd.read_csv(subject_to_photo_path)
    measurements = pd.read_csv(measurements_path)
    hwg_metadata = pd.read_csv(hwg_metadata_path)

    # Merge subject_to_photo_map with measurements
    merged_data = pd.merge(subject_to_photo_map, measurements, on="subject_id", how="inner")
    merged_data = pd.merge(merged_data, hwg_metadata, on="subject_id", how="left")
    print(f"Final dataset size after merging: {len(merged_data)}")
    return merged_data


# Step 2: Dataset Class
class DualSilhouetteDataset(Dataset):
    def __init__(self, data_df, front_dir, side_dir, transform=None):
        self.data_df = data_df
        self.front_dir = front_dir
        self.side_dir = side_dir
        self.transform = transform
        self.measurement_columns = self.data_df.select_dtypes(include=['float32', 'float64', 'int']).columns.tolist()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        front_photo_id = self.data_df.iloc[idx]['photo_id_front']
        side_photo_id = self.data_df.iloc[idx]['photo_id_side']
        front_img_path = os.path.join(self.front_dir, f"{front_photo_id}.png")
        side_img_path = os.path.join(self.side_dir, f"{side_photo_id}.png")
        front_image = Image.open(front_img_path).convert("L")
        side_image = Image.open(side_img_path).convert("L")
        measurements = self.data_df.iloc[idx][self.measurement_columns].values.astype("float32")
        if self.transform:
            front_image = self.transform(front_image)
            side_image = self.transform(side_image)
        return (front_image, side_image), torch.tensor(measurements)


# Step 3: Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Step 4: Define the Model
class DualInputBodyMeasurementModel(nn.Module):
    def __init__(self, num_measurements):
        super(DualInputBodyMeasurementModel, self).__init__()
        self.front_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.side_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.front_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.side_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        combined_features = self.front_model.fc.in_features + self.side_model.fc.in_features
        self.front_model.fc = nn.Identity()
        self.side_model.fc = nn.Identity()
        self.fc = nn.Linear(combined_features, num_measurements)

    def forward(self, front_image, side_image):
        front_features = self.front_model(front_image)
        side_features = self.side_model(side_image)
        combined = torch.cat((front_features, side_features), dim=1)
        return self.fc(combined)


# Step 5: Main Training Code
def main():
    subject_to_photo_path = "Desktop/BodyM-Dataset/train/subject_to_photo_map.csv"
    measurements_path = "Desktop/BodyM-Dataset/train/measurements.csv"
    hwg_metadata_path = "Desktop/BodyM-Dataset/train/hwg_metadata.csv"
    front_dir = "Desktop/BodyM-Dataset/train/mask"
    side_dir = "Desktop/BodyM-Dataset/train/mask_left"
    batch_size = 16

    # Load and merge data
    merged_data = load_and_merge_data(subject_to_photo_path, measurements_path, hwg_metadata_path)
    merged_data['photo_id_front'] = merged_data['photo_id']
    merged_data['photo_id_side'] = merged_data['photo_id']

    # Initialize Dataset and DataLoader
    dataset = DualSilhouetteDataset(data_df=merged_data, front_dir=front_dir, side_dir=side_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with {len(dataset)} samples.")

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualInputBodyMeasurementModel(num_measurements=len(dataset.measurement_columns)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
    total_images = len(dataset)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        image_counter = 0  # Reset image counter for the epoch

        for batch_idx, ((front_images, side_images), measurements) in enumerate(dataloader):
            front_images = front_images.to(device)
            side_images = side_images.to(device)
            measurements = measurements.to(device)

            outputs = model(front_images, side_images)
            if outputs.shape != measurements.shape:
                raise ValueError("Mismatch in shapes of model output and target measurements.")

            loss = criterion(outputs, measurements)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            image_counter += front_images.size(0)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Processed Images: {image_counter}/{total_images} (Batch {batch_idx + 1})")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), "dual_body_measurement_model.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()