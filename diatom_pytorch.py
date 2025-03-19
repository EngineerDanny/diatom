import os
import numpy as np
import pandas as pd
import cv2
import tifffile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(32)  # Use all available CPU cores




class DiatomDataset:
    def __init__(self, csv_path, train_dir, test_dir, target_size=(512, 512)):
        self.csv_path = csv_path
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.target_size = target_size
        
        # Load and process the CSV data
        self.df = pd.read_csv(csv_path)
        
        # Map micrograph IDs to file paths
        self.image_paths = {}
        self.load_image_paths()
        
        # Create augmentation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_image_paths(self):
        # Process training directory
        for file_path in self.train_dir.glob('*.tif'):
            file_name = file_path.name
            # Remove the extension for matching with CSV
            file_name_no_ext = os.path.splitext(file_name)[0]
            self.image_paths[file_name_no_ext] = str(file_path)
        
        # Process test directory
        for file_path in self.test_dir.glob('*.tif'):
            file_name = file_path.name
            # Remove the extension for matching with CSV
            file_name_no_ext = os.path.splitext(file_name)[0]
            self.image_paths[file_name_no_ext] = str(file_path)
        
    def preprocess_image(self, image):
        """Preprocess a single image"""
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def get_xy_data(self, augment=False, test_size=0.2, random_state=42):
        """
        Get processed images and corresponding label counts.
        """
        X = []  # Will hold processed images
        y = []  # Will hold cocconeis counts
        img_ids = []  # To track which images we've processed
        
        # Process each row in the dataframe
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing images"):
            micrograph_id = row['micrograph ID']
            if micrograph_id in self.image_paths:
                # Get cocconeis count
                cocconeis_count = row['Cocconeis']
                
                # Load and preprocess image
                image_path = self.image_paths[micrograph_id]
                image = tifffile.imread(image_path)
                processed_image = self.preprocess_image(image)
                
                # Add to our data
                X.append(processed_image)
                y.append(cocconeis_count)
                img_ids.append(micrograph_id)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have any data
        if len(X) == 0:
            raise ValueError("No matching images found between CSV and image directories")
        
        # Print how many matches were found
        print(f"Found {len(X)} matching images for training")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # If augmentation is requested, augment the training data
        if augment:
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            X_train = np.concatenate([X_train, X_train_aug])
            y_train = np.concatenate([y_train, y_train_aug])
        
        return X_train, X_val, y_train, y_val
    
    def augment_data(self, X, y, augmentation_factor=3):
        X_aug = []
        y_aug = []
        
        for i in tqdm(range(len(X)), desc="Augmenting data"):
            image = X[i]
            count = y[i]
            
            # Create multiple augmented versions of each image
            for _ in range(augmentation_factor):
                # Transform to tensor and back to numpy for each augmentation
                tensor_image = self.transform(image)
                # Convert tensor [C,H,W] back to numpy [H,W,C]
                aug_image = tensor_image.permute(1, 2, 0).numpy()
                X_aug.append(aug_image)
                y_aug.append(count)
        
        return np.array(X_aug), np.array(y_aug)
    
    def create_augmented_dataset(self, output_dir, augmentation_factor=3):
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create a CSV to track augmented images
        aug_data = []
        
        # Process each row in the dataframe
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating augmented dataset"):
            micrograph_id = row['micrograph ID']
            if micrograph_id in self.image_paths:
                # Get original image and cocconeis count
                image_path = self.image_paths[micrograph_id]
                cocconeis_count = row['Cocconeis']
                
                # Load image
                image = tifffile.imread(image_path)
                
                # Copy original image to augmented folder
                original_output_path = output_path / micrograph_id
                shutil.copy(image_path, original_output_path)
                
                # Add original to CSV
                aug_data.append({
                    'image_id': micrograph_id,
                    'original_image': micrograph_id,
                    'cocconeis_count': cocconeis_count,
                    'augmented': False
                })
                
                # Create augmented versions
                for aug_idx in range(1, augmentation_factor + 1):
                    
                    # Create augmented image
                    tensor_image = transforms.ToTensor()(image)
                    tensor_image = transforms.RandomHorizontalFlip(p=0.5)(tensor_image)
                    tensor_image = transforms.RandomVerticalFlip(p=0.5)(tensor_image)
                    tensor_image = transforms.RandomRotation(90)(tensor_image)
                    tensor_image = transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    )(tensor_image)
                    # Convert back to numpy array for saving [C,H,W] -> [H,W,C]
                    aug_image = tensor_image.permute(1, 2, 0).numpy() * 255.0
                    aug_image = aug_image.astype(np.uint8)
                    
                    # Create filename for augmented image
                    base_name, ext = os.path.splitext(micrograph_id)
                    aug_filename = f"{base_name}_aug{aug_idx}{ext}"
                    aug_output_path = output_path / aug_filename
                    
                    # Save augmented image
                    tifffile.imwrite(aug_output_path, aug_image)
                    
                    # Add to CSV
                    aug_data.append({
                        'image_id': aug_filename,
                        'original_image': micrograph_id,
                        'cocconeis_count': cocconeis_count,
                        'augmented': True
                    })
        
        # Save augmentation metadata
        aug_df = pd.DataFrame(aug_data)
        aug_df.to_csv(output_path / "augmentation_metadata.csv", index=False)
        
        return aug_df


class DiatomDatasetTorch(Dataset):
    """
    PyTorch Dataset for diatom images and count labels.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H,W,C) -> (C,H,W)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DiatomCountModel(nn.Module):
    """
    PyTorch model for diatom counting using transfer learning.
    """
    def __init__(self):
        super(DiatomCountModel, self).__init__()
        
        # Use pre-trained ResNet18 as base model
        # EfficientNet would be better but requires additional libraries
        self.base_model = models.resnet18(pretrained=True)
        
        # Replace final layer with a regression head
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove classifier
        
        # Create regression head
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.ReLU()  # ReLU ensures non-negative count
        )
    
    def forward(self, x):
        features = self.base_model(x)
        count = self.regression_head(features)
        return count


class DiatomTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, save_dir='models'):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train for
            lr: Learning rate
            save_dir: Directory to save model
            
        Returns:
            Dictionary with training history
        """
        # Create directory for saving models
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': []
        }
        
        # Best validation loss
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)  # Reshape to match output
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item() * images.size(0)
            
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    # Move data to device
                    images = images.to(self.device)
                    labels = labels.to(self.device).view(-1, 1)  # Reshape to match output
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    # Accumulate loss
                    val_loss += loss.item() * images.size(0)
                    
                    # Store predictions and labels for metrics
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate average validation loss and MAE
            val_loss /= len(val_loader.dataset)
            val_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'diatom_counter_best.pth'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'diatom_counter_final.pth'))
        
        return history
    
    def predict(self, data_loader):
        """
        Make predictions with the model.
        
        Args:
            data_loader: DataLoader with images to predict on
            
        Returns:
            Numpy array with predictions
        """
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="Predicting"):
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Store predictions
                all_preds.extend(outputs.cpu().numpy())
        
        # Convert to numpy array and round to integers
        predictions = np.array(all_preds).reshape(-1)
        predictions_rounded = np.round(predictions).astype(int)
        
        return predictions_rounded
    
    def evaluate(self, data_loader):
        """
        Evaluate model performance.
        
        Args:
            data_loader: DataLoader with images and labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        predictions = self.predict(data_loader)
        
        # Get ground truth labels
        all_labels = []
        for _, labels in data_loader:
            all_labels.extend(labels.numpy())
        
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - all_labels))
        mse = np.mean((predictions - all_labels) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate exact match accuracy (predictions match exactly)
        exact_match = np.mean(predictions == all_labels)
        
        # Calculate accuracy within 1 count
        within_one = np.mean(np.abs(predictions - all_labels) <= 1)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'exact_match': exact_match,
            'within_one': within_one
        }
        
        return metrics
    
    def visualize_predictions(self, data_loader, n_samples=5):

        # Get a batch of data
        images, labels = next(iter(data_loader))
        images = images.to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            predictions = outputs.cpu().numpy().reshape(-1)
            predictions_rounded = np.round(predictions).astype(int)
        
        # Select samples to visualize
        n_samples = min(n_samples, len(images))
        indices = np.random.choice(range(len(images)), size=n_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        # Plot each sample
        for i, idx in enumerate(indices):
            # Get image, true count, and predicted count
            image = images[idx].cpu().permute(1, 2, 0).numpy()  # Convert to numpy and channels last
            true_count = labels[idx].item()
            pred_count = predictions_rounded[idx]
            
            # Calculate error
            error = pred_count - true_count
            
            # Plot image
            axes[i].imshow(image)
            axes[i].set_title(f"True: {true_count}, Predicted: {pred_count}, Error: {error}")
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    
    
"""Main execution function"""
# Define paths
csv_path = "/projects/genomic-ml/da2343/diatom/Kraken_2023_measurements.csv"
train_dir = "/projects/genomic-ml/da2343/diatom/train/"
test_dir = "/projects/genomic-ml/da2343/diatom/test/"
augmented_dir = "/projects/genomic-ml/da2343/diatom/augmented/"
models_dir = "/projects/genomic-ml/da2343/diatom/models/"

# Create dataset object
print("Initializing dataset...")
diatom_dataset = DiatomDataset(csv_path, train_dir, test_dir)

# # Create augmented dataset
# print("Creating augmented dataset...")
# aug_df = diatom_dataset.create_augmented_dataset(
#     augmented_dir, 
#     augmentation_factor=5  # Create 5 augmented versions per image
# )
# print(f"Created {len(aug_df)} total images (original + augmented)")

# Get processed data
print("Processing data for model training...")
X_train, X_val, y_train, y_val = diatom_dataset.get_xy_data(
    augment=False,
    test_size=0.2
)
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Validation data: {X_val.shape}, {y_val.shape}")

# Create PyTorch datasets
train_dataset = DiatomDatasetTorch(X_train, y_train)
val_dataset = DiatomDatasetTorch(X_val, y_val)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True,
    num_workers=16
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=16
)

# Create model
print("Creating model...")
model = DiatomCountModel()

# Create trainer
trainer = DiatomTrainer(model)

# Train model
print("Training model...")
history = trainer.train(
    train_loader, 
    val_loader, 
    epochs=40,
    lr=0.0001,
    save_dir=models_dir
)

# Evaluate model
print("Evaluating model...")
metrics = trainer.evaluate(val_loader)
print("Evaluation metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Visualize some predictions
print("Visualizing predictions...")
fig = trainer.visualize_predictions(val_loader, n_samples=5)
fig.savefig(os.path.join(models_dir, "prediction_visualization.png"))

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(models_dir, "training_history.png"))

print("Diatom counting model training complete!")
