import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from model import build_model

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])

dataset = datasets.ImageFolder(

    "dataset/cnn_dataset",

    transform=transform

)

print(
    "Classes:",
    dataset.classes
)

print(
    "Total Images:",
    len(dataset)
)

train_loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)

model = build_model(

    len(dataset.classes)

).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4

)

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(
            images
        )

        loss = loss_fn(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(
            1
        )

        correct += (
            preds==labels
        ).sum().item()

        total += labels.size(0)

    acc = 100*correct/total

    print(

        f"Epoch {epoch+1}/{epochs}",

        f"Loss:{running_loss:.3f}",

        f"Acc:{acc:.2f}%"

    )

torch.save(

    model.state_dict(),

    "blood_cnn.pth"

)

print()

print("Training completed")

print("Saved: blood_cnn.pth")