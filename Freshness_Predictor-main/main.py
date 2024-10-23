# app.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import torchmetrics
from PIL import Image
import streamlit as st
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Fruit Freshness Predictor",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title("🍎 Fruit Freshness Predictor")

# Description
st.write("""
This application predicts the type of fruit and its freshness (Fresh or Stale) based on the uploaded image.
""")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the ModelVGG16 class
class ModelVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        
        self.base = models.vgg16(pretrained=True)
        
        # Freeze all layers except the last 15
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
                    
        self.base.classifier = nn.Sequential()  # Clear classifier
        self.base.fc = nn.Sequential()  # Remove fc layers
            
        # Custom blocks
        self.block1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),  # Adjust input size based on VGG16 output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)  # Assuming 9 classes of fruits
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # Fresh or Stale (binary classification)
        )

        # Loss function
        self.loss_fxn = nn.CrossEntropyLoss()

        # Accuracy metrics
        self.fruit_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=9)
        self.fresh_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2)

    def forward(self, x):
        x = self.base.features(x)  # Use VGG16's convolutional layers
        x = torch.flatten(x, 1)    # Flatten the output
        x = self.block1(x)         # Pass through custom block1
        y1, y2 = self.block2(x), self.block3(x)  # Get predictions from block2 and block3
        return y1, y2

# Function to load the model with caching to prevent reloading on every run
@st.cache_resource
def load_model(model_path):
    model = ModelVGG16().to(device)
    try:
        # Load the state_dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        st.success(f"Model loaded successfully from `{model_path}`")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    return model

# Load the model
model_path = "model.pth"  # Ensure 'model.pth' is in the same directory or provide the full path
model_vgg16 = load_model(model_path)

# Class names (ensure this matches your training)
class_names = [
    'apple', 'banana', 'orange', 'strawberry', 'tomato',
    'grape', 'pineapple', 'mango', 'blueberry'  # Add up to 9 classes
]

# Function to preprocess the image
def preprocess_image(image):
    # Convert PIL Image to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(image).unsqueeze(0)  # Add batch dimension
    return img_t

# Function to run the prediction
def predict_freshness(image):
    image = preprocess_image(image)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model_vgg16(image)
        # Assuming the model has two outputs for fruit type and freshness
        fruit_pred = torch.argmax(outputs[0], axis=1).cpu().numpy()[0]
        fresh_pred = torch.argmax(outputs[1], axis=1).cpu().numpy()[0]
    
    # Map the predictions to labels
    fruit_label = class_names[fruit_pred] if fruit_pred < len(class_names) else "Unknown"
    freshness_label = 'Fresh' if fresh_pred == 0 else 'Stale'
    
    return fruit_label, freshness_label

# File uploader allows user to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Run prediction
        fruit_label, freshness_label = predict_freshness(image)
        
        # Display the results
        st.success(f"**Prediction:** {fruit_label}, {freshness_label}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to get started.")

# Footer
st.write("---")
st.write("Developed by [Your Name](https://your-website.com) | © 2024")
