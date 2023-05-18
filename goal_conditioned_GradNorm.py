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
import cv2
import torch.nn.functional as F
from scipy.spatial import distance

num_classes = 4

def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    dist = np.dot(np.dot(diff, cov_inv), diff.T)
    return dist

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

class getModel(nn.Module):
    def __init__(self):
        super(getModel, self).__init__()

        # image branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Distance Branch
        self.distance_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.combined_branch = nn.Sequential(
            nn.Linear(23104 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 4)
        )

    def forward(self, x_images, x_distances):
        x1 = self.image_branch(x_images)
        x2 = self.distance_branch(x_distances)
        x_combined = torch.cat([x1, x2], dim=1)
        
        action_output = self.combined_branch(x_combined)

        self.features = action_output

        # Calculate the softmax probabilities
        action_probabilities = F.softmax(action_output, dim=1)
        
        # Compute the entropy
        entropy = -torch.sum(action_probabilities * torch.log(action_probabilities), dim=1)
        print("Action Probabilities:", action_probabilities)
        print("Entropy:", entropy)
        return action_output, x1

def train_model(X_images, X_distances, y):
    X_images_train, X_images_test, X_distances_train, X_distances_test, y_train, y_test = train_test_split(X_images, X_distances, y, test_size=0.2, random_state=42)

    train_dataset = mixedDataset(X_images_train, X_distances_train, y_train)
    test_dataset = mixedDataset(X_images_test, X_distances_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = getModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []
    test_losses = []
    train_features = []
    train_labels = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for img, dist, label in train_loader:
            
            optimizer.zero_grad()
            action_output, _ = model(img, dist)
            loss = criterion(action_output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_features.append(model.features.detach().numpy())
            train_labels.append(label.numpy())

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, dist, label in test_loader:
                action_output, _ = model(img, dist)
                loss = criterion(action_output, label)
                test_loss += loss.item()
                _, predicted = torch.max(action_output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        accuracy = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')


    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)

    class_means = []
    class_covs = []

    for i in range(num_classes):
        class_features = train_features[train_labels == i]
        class_mean = class_features.mean(axis=0)
        class_cov = np.cov(class_features, rowvar=False)
        class_cov_inv = np.linalg.pinv(class_cov)
        class_means.append(class_mean)
        class_covs.append(class_cov_inv)

    
    # Plot
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def ood_detection_gradnorm(model, img, dist, temperature=1, num_classes=4):
    model.eval()
    img.unsqueeze_(0)
    dist.unsqueeze_(0)

    # Enable gradients for intermediate features
    img.requires_grad = True
    dist.requires_grad = True

    action_output, local_output = model(img, dist)
    action_output = torch.softmax(action_output / temperature, dim=1)

    # Calculate the gradient norm
    grad_outputs = torch.zeros_like(local_output)
    for i in range(num_classes):
        grad_outputs[:, i] = 1
        grads = torch.autograd.grad(outputs=local_output, inputs=img, grad_outputs=grad_outputs, create_graph=True)[0]
        if i == 0:
            grad_norms = grads.norm(2, dim=(1, 2, 3)).unsqueeze(1)
        else:
            grad_norms = torch.cat((grad_norms, grads.norm(2, dim=(1, 2, 3)).unsqueeze(1)), dim=1)
        grad_outputs[:, i] = 0

    ood_score = (grad_norms * action_output).sum(dim=1).item()
    # The higher the ood_score, the more likely the input is OOD. You can set a threshold for classification.
    return ood_score

def ood_score_mahalanobis(model, img, dist, class_means, class_covs):
    model.eval()

    action_output, _ = model(img, dist)
    test_feature = model.features.detach().numpy()

    ood_scores = []
    for i in range(len(class_means)):
        mean = class_means[i]
        cov_inv = class_covs[i]
        dist = mahalanobis_distance(test_feature, mean, cov_inv)
        ood_scores.append(dist)

    return np.min(ood_scores)
# Beginning of the code
with open('images_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

X_images, X_distances, y = preprocess_data(data)
"""
### Train the model
model = train_model(X_images, X_distances, y)

# Save the model
torch.save(model.state_dict(), 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/model_weights_GradNorm.pth')
"""
### Load the model weights
model = getModel()
model_weights_path = 'C:/Users/moham/Desktop/UMD/Spring 2023/CMSC828A/Final Project/IROS_codes/IROS_codes/model_weights_GradNorm.pth'
model.load_state_dict(torch.load(model_weights_path))




"""

########################
#
# Test GradNorm
#
########################
X_images_train, X_images_test, X_distances_train, X_distances_test, y_train, y_test = train_test_split(X_images, X_distances, y, test_size=0.2, random_state=42)

# create train and test dataset
train_dataset = mixedDataset(X_images_train, X_distances_train, y_train)
test_dataset = mixedDataset(X_images_test, X_distances_test, y_test)

# OOD detection
for i in range(1):
    img, dist, _ = test_dataset[20]
    ood_score = ood_detection_gradnorm(model, img, dist)
    print(f"OOD Score: {ood_score}")

    # Set a threshold for OOD classification
    threshold = 0.5  # Replace this with the desired threshold
    is_ood = ood_score > threshold
    print(f"Is OOD: {is_ood}")


"""



def render(matrix, name):
    cmap = plt.cm.GnBu
    cmap.set_over('black', alpha=.85)
    plt.figure(figsize=(5,5))
    plt.imshow(matrix, cmap=cmap, vmin=0, vmax=20)
    plt.axis("off")
    plt.axis("tight")
    plt.axis("image")
    plt.savefig("images_tests/" + name + ".png", dpi=20, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 0

def create_test_datapoint(input_list):
    distance_to_goal = input_list[:2]
    matrix = np.array(input_list[2:]).reshape(5, 5)

    # Render the matrix as an image
    image_name = "local_image_test2"
    render(matrix, image_name)

    # Read the local image
    local_image_path = os.path.join("images_tests/", image_name + ".png")
    local_img = cv2.imread(local_image_path, cv2.IMREAD_UNCHANGED)
    local_img = np.transpose(local_img, (2, 0, 1))
    local_img = local_img.astype(np.float32)
    local_img = torch.tensor(local_img)

    distance_to_goal = np.array(distance_to_goal, dtype=np.float32)
    distance_to_goal = torch.tensor(distance_to_goal)

    return local_img, distance_to_goal

# Define your input list with 17 elements (first two elements are the distances to the goal, next 15 elements are for the 5x5 matrix)
input_list = [ 20. , 0. , 30., 30. ,30., 30., 30.,  10.,  0. , 0. , 30. , 0. , 0. , 30. ,10.,  30. , 30. , 0. ,0. , 0.,  0.,  0. , 0. , 0. , 0. , 0. , 0.]
input_list_id = [ 0. , 3. , 30., 30. ,30., 30., 30.,  0.,  0. , 0. , 0. , 0. , 0. , 0. ,10.,  0. , 0. , 0. ,0. , 0.,  0.,  0. , 0. , 0. , 0. , 0. , 0.]
               

# Create a test datapoint
local_img, dist = create_test_datapoint(input_list)

# OOD detection
ood_score = ood_detection_gradnorm(model, local_img, dist)
print(f"OOD Score for the test image: {ood_score}")

# Set a threshold for OOD classification
threshold = 0.5  
is_ood = ood_score > threshold
print(f"Is OOD: {is_ood}")



ood_score = ood_score_mahalanobis(model, local_img, dist)
print(f"OOD Score for the test image: {ood_score}")

threshold = 0.5
is_ood = ood_score > threshold
print(f"Is OOD: {is_ood}")

