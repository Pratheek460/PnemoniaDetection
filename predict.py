import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
from src.gradcam import generate_heatmap
from PIL import Image

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = models.densenet121()

model.classifier = nn.Linear(
    model.classifier.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/pneumonia_model.pth",
        map_location=device
    )
)

model.eval()

classes = [
    "NORMAL",
    "PNEUMONIA"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(path):

    image = Image.open(path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probs = F.softmax(
            output,
            dim=1
        )

        confidence,pred = torch.max(
            probs,
            1
        )

    heatmap_path = (
        f"heatmaps/{pred.item()}_{confidence.item()}.jpg"
    )

    generate_heatmap(
        model,
        image,
        path,
        heatmap_path
    )

    return {

        "prediction":
        classes[pred.item()],

        "confidence":
        round(
            confidence.item()*100,
            2
        ),

        "heatmap":
        heatmap_path
    }