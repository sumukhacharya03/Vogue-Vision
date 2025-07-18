import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
from io import BytesIO

class MultiOutputFashionModel(nn.Module):
    def __init__(self, num_genders, num_types, num_seasons, num_colors):
        super(MultiOutputFashionModel, self).__init__()

        self.backbone = models.resnet50()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Output heads for each task
        self.gender_head = nn.Linear(num_ftrs, num_genders)
        self.type_head = nn.Linear(num_ftrs, num_types)
        self.season_head = nn.Linear(num_ftrs, num_seasons)
        
        self.colour_head = nn.Linear(num_ftrs, num_colors) 

    def forward(self, x):
        features = self.backbone(x)
        
        return {
            'gender': self.gender_head(features),
            'masterCategory': self.type_head(features),
            'season': self.season_head(features),
            'baseColour': self.colour_head(features)
        }

device = torch.device("cpu")

try:
    with open('label_mappings.json', 'r') as f:
        label_mappings = json.load(f)
except FileNotFoundError:
    st.error("Error: 'label_mappings.json' not found. Please make sure the file is in the same directory as app.py.")
    st.stop()


num_classes = {col: len(label_mappings[col]) for col in label_mappings}

model = MultiOutputFashionModel(
    num_genders=num_classes['gender'],
    num_types=num_classes['masterCategory'],
    num_seasons=num_classes['season'],
    num_colors=num_classes['baseColour']
)

try:
    model.load_state_dict(torch.load('best_fashion_model.pth', map_location=device))
except FileNotFoundError:
    st.error("Error: 'best_fashion_model.pth' not found. Please make sure the file is in the same directory as app.py.")
    st.stop()

model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Fashion Product Predictor", layout="centered")
st.title("Fashion Product Attribute Predictor")
st.write("Upload an image of a fashion item, and the model will predict its attributes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    
    if st.button('Predict Attributes'):
        with st.spinner('Analyzing the image...'):
            transformed_image = data_transforms(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(transformed_image)
            
            predictions = {}
            for task, out in outputs.items():
                _, pred_idx = torch.max(out, 1)
                pred_label = label_mappings[task][str(pred_idx.item())]
                predictions[task] = pred_label

        # Prediction Results
        st.subheader("Prediction Results:")
        st.success(f"**Gender:** {predictions['gender']}")
        st.success(f"**Product Type:** {predictions['masterCategory']}")
        st.success(f"**Season:** {predictions['season']}")
        st.success(f"**Base Color:** {predictions['baseColour']}")
