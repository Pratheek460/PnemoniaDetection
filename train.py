import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

print("Starting training...")

train_dataset = datasets.ImageFolder(
    "data/chest_xray/train",
    transform=transform
)

print("Train dataset loaded.")
print("Number of training images:", len(train_dataset))

test_dataset = datasets.ImageFolder(
    "data/chest_xray/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

model = models.densenet121(
    weights=None
)

for param in model.parameters():
    param.requires_grad=False

model.classifier = nn.Linear(
    model.classifier.in_features,
    2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

for epoch in range(5):

    model.train()

    for images,labels in train_loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}")

torch.save(
    model.state_dict(),
    "models/pneumonia_model.pth"
)