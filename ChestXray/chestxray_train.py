import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from chestxray_model import ChestXrayCNN

def load_data(batch_size=64):
    print("Step 1: Loading data...")
    
    # Define the absolute paths
    base_path = os.path.abspath('./chestxray_data')
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomVerticalFlip(),    # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaded. Training size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, test_loader

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20, patience=3):
    print("Step 2: Starting training...")
    model.train()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = 'chestxray_best_model.pth'
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation
        print("Validating...")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total

        print(f'  Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model with the best validation loss
            torch.save(model.state_dict(), best_model_path)
            print("  Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("  Early stopping due to no improvement in validation loss.")
                break
        
        # Step the learning rate scheduler
        scheduler.step()
        
        model.train()  # Return model to training mode after evaluation

def save_model(model, path='chestxray_cnn.pth'):
    print(f"Step 3: Saving model to {path}...")
    torch.save(model.state_dict(), path)
    if os.path.exists(path):
        print(f"Model saved successfully to {path}")
    else:
        print("Model saving failed!")

if __name__ == '__main__':
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXrayCNN().to(device)
    
    train_loader, val_loader = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler
    
    print("Starting the training process...")
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    print("Training completed.")
    
    print("Saving the final model...")
    save_model(model)
