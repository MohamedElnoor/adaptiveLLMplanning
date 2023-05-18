import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Class for the image dataset
class ImageDataset(Dataset):
    def __init__(self, X_global_images, X_local_images, y):
        self.X_global_images = X_global_images
        self.X_local_images = X_local_images
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_global_images[idx], self.X_local_images[idx], self.y[idx]

def preprocess_data(data):
    global_image_paths, local_image_paths, actions = zip(*data)
    X_global_images = np.array([plt.imread(img_path) for img_path in global_image_paths])
    X_local_images = np.array([plt.imread(img_path) for img_path in local_image_paths])
    y = np.array(actions)

    X_global_images = torch.tensor(X_global_images, dtype=torch.float32).permute(0, 3, 1, 2)
    X_local_images = torch.tensor(X_local_images, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long)

    return X_global_images, X_local_images, y

class getModel(nn.Module):
    def __init__(self):
        super(getModel, self).__init__()

        # global image branch
        self.global_image_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 64), # Updated input size
            nn.ReLU(),
        )

        # local image branch
        self.local_image_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 32), # Updated input size
            nn.ReLU(),
        )

        # Merge layer
        self.merge_layer = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x_global_images, x_local_images):
        global_output = self.global_image_branch(x_global_images)
        local_output = self.local_image_branch(x_local_images)
        combined_output = torch.cat((global_output, local_output), dim=1)
        output = self.merge_layer(combined_output)
        return output, local_output

def train_model(X_global_images, X_local_images, y):
    # Split the dataset
    X_global_images_train, X_global_images_test, X_local_images_train, X_local_images_test, y_train, y_test = train_test_split(X_global_images, X_local_images, y, test_size=0.2, random_state=42)

    # create train and test dataset
    train_dataset = ImageDataset(X_global_images_train, X_local_images_train, y_train)
    test_dataset = ImageDataset(X_global_images_test, X_local_images_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = getModel()

    criterion = nn.CrossEntropyLoss() # for loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for global_img, local_img, label in train_loader:
            optimizer.zero_grad()
            output = model(global_img, local_img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for global_img, local_img, label in test_loader:
                output = model(global_img, local_img)
                loss = criterion(output, label)
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        accuracy = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2, marker='o', markersize=5)
    plt.plot(test_losses, label='Validation Loss', linewidth=2, marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Training and Validation Losses", fontsize=16)
    plt.show()

    return model


def ood_detection_gradnorm(model, global_img, local_img, temperature=1, num_classes=4):
    model.eval()
    global_img.unsqueeze_(0)
    local_img.unsqueeze_(0)

    # Enable gradients for intermediate features
    global_img.requires_grad = True
    local_img.requires_grad = True
    
    output, local_output = model(global_img, local_img)
    output = torch.softmax(output / temperature, dim=1)

    # Calculate the gradient norm
    grad_outputs = torch.zeros_like(local_output)
    for i in range(num_classes):
        grad_outputs[:, i] = 1
        grads = torch.autograd.grad(outputs=local_output, inputs=local_img, grad_outputs=grad_outputs, create_graph=True)[0]
        if i == 0:
            grad_norms = grads.norm(2, dim=(1, 2, 3)).unsqueeze(1)
        else:
            grad_norms = torch.cat((grad_norms, grads.norm(2, dim=(1, 2, 3)).unsqueeze(1)), dim=1)
        grad_outputs[:, i] = 0

    ood_score = (grad_norms * output).sum(dim=1).item()
    return ood_score

#### Begining of the code

with open('images_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

X_global_images, X_local_images, y = preprocess_data(data)


"""
### Train the model
model = train_model(X_global_images, X_local_images, y)

# Replace the path to your local directory
torch.save(model.state_dict(), 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/model_weights_local_and_global_images.pth')
"""

### Load the model weights

model = getModel()
model_weights_path = 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/model_weights_local_and_global_images.pth'
model.load_state_dict(torch.load(model_weights_path))




########################
#
# Test GradNorm
#
########################

X_global_images_train, X_global_images_test, X_local_images_train, X_local_images_test, y_train, y_test = train_test_split(X_global_images, X_local_images, y, test_size=0.2, random_state=42)

# create train and test dataset
train_dataset = ImageDataset(X_global_images_train, X_local_images_train, y_train)
test_dataset = ImageDataset(X_global_images_test, X_local_images_test, y_test)

# OOD detection
for i in range(10):
    global_img, local_img, _ = test_dataset[0]
    ood_score = ood_detection_gradnorm(model, global_img, local_img)
    print(f"OOD Score: {ood_score}")

    # Set a threshold for OOD classification
    threshold = 0.5  # Replace this with the desired threshold
    is_ood = ood_score > threshold
    print(f"Is OOD: {is_ood}")

### Test OOD from home dir

global_img_path = 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/global_image_1_1.png'
local_img_path = 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/local_image_1_1.png'

# Load the images using cv2
global_img = cv2.imread(global_img_path, cv2.IMREAD_UNCHANGED)
local_img = cv2.imread(local_img_path, cv2.IMREAD_UNCHANGED)

# Transform the images into tensors
def transform_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255
    img = torch.tensor(img)
    return img

global_img = transform_image(global_img)
local_img = transform_image(local_img)

ood_score = ood_detection_gradnorm(model, global_img, local_img)
print(f"OOD Score: {ood_score}")

# Set a threshold for OOD classification
threshold = 0.5  # Replace this with the desired threshold
is_ood = ood_score > threshold
print(f"Is OOD: {is_ood}")