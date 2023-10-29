import torch
from PIL import Image, ImageOps
from torchvision import transforms
from ModelTrainer import ShapeClassifier

# Load the trained model
model = ShapeClassifier(num_classes=7)  # Replace with the correct number of classes
model.load_state_dict(torch.load('shape_classifier.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess the new grayscale image
new_image = Image.open('img.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for grayscale
])
new_image = transform(new_image).unsqueeze(0)  # Add a batch dimension


# Make predictions
with torch.no_grad():
    output = model(new_image)

# Convert the model's output to a binary format (e.g., using a threshold)
threshold = 0.5  # You can adjust the threshold as needed
predicted_labels = (output > threshold).int().tolist()[0]

print("Predicted labels:", predicted_labels)






