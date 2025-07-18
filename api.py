import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
from io import BytesIO

class MultiOutputFashionModel(nn.Module):
    def __init__(self, num_genders, num_types, num_seasons, num_colours):
        super(MultiOutputFashionModel, self).__init__()
        self.backbone = models.resnet50()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.gender_head = nn.Linear(num_ftrs, num_genders)
        self.type_head = nn.Linear(num_ftrs, num_types)
        self.season_head = nn.Linear(num_ftrs, num_seasons)
        self.colour_head = nn.Linear(num_ftrs, num_colours)
    def forward(self, x):
        features = self.backbone(x)
        return {
            'gender': self.gender_head(features),
            'masterCategory': self.type_head(features),
            'season': self.season_head(features),
            'baseColour': self.colour_head(features)
        }

device = torch.device("cpu")
with open('label_mappings.json', 'r') as f:
    label_mappings = json.load(f)

num_classes = {col: len(label_mappings[col]) for col in label_mappings}
model = MultiOutputFashionModel(
    num_genders=num_classes['gender'],
    num_types=num_classes['masterCategory'],
    num_seasons=num_classes['season'],
    num_colours=num_classes['baseColour']
)
model.load_state_dict(torch.load('best_fashion_model.pth', map_location=device))
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    
    transformed_image = data_transforms(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(transformed_image)
        
    predictions = {}
    for task, out in outputs.items():
        _, pred_idx = torch.max(out, 1)
        pred_label = label_mappings[task][str(pred_idx.item())]
        predictions[task] = pred_label
        
    return predictions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
