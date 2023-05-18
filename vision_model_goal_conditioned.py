import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Class for the mixed dataset (mixed = image + distance to goal list)
class mixedDataset(Dataset):
    def __init__(self, X_images, X_distances, y):
        self.X_images = X_images
        self.X_distances = X_distances
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_images[idx], self.X_distances[idx], self.y[idx]

def preprocess_data(data):

    image_paths, distances, actions = zip(*data)
    X_images = np.array([plt.imread(img_path) for img_path in image_paths])
    X_distances = np.array(distances)
    y = np.array(actions)

    X_images = torch.tensor(X_images, dtype=torch.float32).permute(0, 3, 1, 2)
    X_distances = torch.tensor(X_distances, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X_images, X_distances, y


# The model 

class getModel(nn.Module):
    def __init__(self):
        super(getModel, self).__init__()

        # image branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Distance Branch
        self.distance_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.combined_branch = nn.Sequential(
            nn.Linear(23104 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x_images, x_distances):
        x1 = self.image_branch(x_images)
        #print("test",x1.shape)
        x2 = self.distance_branch(x_distances)
        x = torch.cat([x1, x2], dim=1)
        return self.combined_branch(x)

def train_model(X_images, X_distances, y):

    # Spilt the dataset
    X_images_train, X_images_test, X_distances_train, X_distances_test, y_train, y_test = train_test_split(X_images, X_distances, y, test_size=0.2, random_state=42)

    # create train and test dataset
    train_dataset = mixedDataset(X_images_train, X_distances_train, y_train)
    test_dataset = mixedDataset(X_images_test, X_distances_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = getModel()

    criterion = nn.CrossEntropyLoss() # for loss
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    num_epochs = 10
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for img, dist, label in train_loader:
            optimizer.zero_grad()
            output = model(img, dist)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, dist, label in test_loader:
                output = model(img, dist)
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
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model






#### Besgining of the code

with open('images_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

X_images, X_distances, y = preprocess_data(data)

model = train_model(X_images, X_distances, y)

# Replace the path to your local directory
torch.save(model.state_dict(), 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/model_weights.pth')
