import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Define a custom dataset class
class FreshnessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Define transformations for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#loading datasets
def load_data(folder_path):
    image_paths = []
    labels = []
    
    label_map = {
        'freshapples': 0,
        'freshbanana': 0,
        'freshoranges': 0,
        'rottenapples': 1,
        'rottenbanana': 1,
        'rottenoranges': 1,
    }

    for folder_name in os.listdir(folder_path):
        if folder_name in label_map:
            folder_path_full = os.path.join(folder_path, folder_name)
            for filename in os.listdir(folder_path_full):
                if filename.endswith('.jpg') or filename.endswith('.png'):  # Include other formats if needed
                    image_paths.append(os.path.join(folder_path_full, filename))
                    labels.append(label_map[folder_name])

    return image_paths, labels

train_image_paths, train_labels = load_data('dataset/train')
test_image_paths, test_labels = load_data('dataset/test')

# dataloaders
train_dataset = FreshnessDataset(train_image_paths, train_labels, transform=transform)
test_dataset = FreshnessDataset(test_image_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

#cnn model (resnet)
class FreshnessModel(nn.Module):
    def __init__(self):
        super(FreshnessModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True) 
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Change output layer to match classes

    def forward(self, x):
        return self.base_model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FreshnessModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop with detailed progress
num_epochs = 5
loss_values = [] 
best_accuracy = 0.0

def validate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

print("Starting training...")
print(f"Training on device: {device}")
print(f"Total batches per epoch: {len(train_loader)}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    
    # progress bar for batches
    progress_bar = tqdm(enumerate(train_loader), 
                       total=len(train_loader),
                       desc=f'Epoch {epoch + 1}/{num_epochs}',
                       leave=True)
    
    for batch_idx, (images, labels) in progress_bar:
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix({
            'batch': f'{batch_idx + 1}/{len(train_loader)}',
            'loss': f'{loss.item():.4f}'
        })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)
    epoch_time = time.time() - epoch_start_time
    
    # Validate the model
    val_accuracy, val_loss = validate_model(model, test_loader, criterion, device)
    print(f'\nEpoch {epoch + 1}/{num_epochs} Summary:')
    print(f'Training Loss: {epoch_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Time taken: {epoch_time:.2f} seconds')
    
    # Save model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'freshness_model.pth')
        print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
    
    print('-' * 60)

print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print(f"Model saved as 'freshness_model.pth'")

# Plot with details
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Training Loss Over Epochs', fontsize=14, pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, num_epochs + 1))

# Add loss values on points
for i, loss in enumerate(loss_values):
    plt.annotate(f'{loss:.4f}', 
                (i + 1, loss),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center')

plt.tight_layout()
plt.savefig('training_loss_graph.png', dpi=300, bbox_inches='tight')
print("Enhanced loss graph saved as 'training_loss_graph.png'")

# Evaluation (simplified)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')