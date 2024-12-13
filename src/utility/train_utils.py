import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


state_map = {
    "husky": {
        "sleeping": 0, 
        "eating": 1, 
        "playing": 2, 
        "sitting": 3
    },
}

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, transform=None, dataset="husky"):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = state_map[dataset]


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # Assuming the image path is in the first column
        caption = self.dataframe.iloc[idx, 1]   # Assuming the caption is in the second column
        state = self.dataframe.iloc[idx, 2]   # Assuming the status is in the second column

        # Load the image
        image = Image.open(img_path)
        
        # Apply the specified transformations to the image if provided
        if self.transform:
            image = self.transform(image)
        
        # Return the image and caption as a tuple
        return image, caption, self.label_map[state]
    

class MobileNetCaptioning(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetCaptioning, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)  # Replace the classifier for your task

    def forward(self, x):
        return self.mobilenet(x)


def train(model, dataloader, num_epochs, learning_rate, val_dataloader, version):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        model.train()

        for images, _, states in tqdm(dataloader):
            images, states = images.to(device), states.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, states)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += states.size(0)
            correct_predictions += (predicted == states).sum().item()

        epoch_accuracy = 100 * correct_predictions / total_samples

        val_acc = evaluate(model, val_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader)}. Train acc: {epoch_accuracy:.2f}%. Val acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, f'./saved_models/mobilenet_{version}.pth')

    print("Training finished!")


def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, _, states in dataloader:
            images, states = images.to(device), states.to(device)
            

            # Forward pass
            outputs = model(images)
            # Calculate predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Update counts
            total_samples += states.size(0)
            correct_predictions += (predicted == states).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    return accuracy