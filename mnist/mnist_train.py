import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from mnist_model import CNNModel
import os

def load_data(batch_size=64):
    print("Step 1: Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the MNIST dataset
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders for training and validation
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaded. Training size: {train_size}, Validation size: {val_size}")
    return train_loader, val_loader

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, patience=3):
    print("Step 2: Starting training...")
    model.train()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
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
            torch.save(model.state_dict(), 'mnist_cnn_best.pth')
            print("  Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("  Early stopping due to no improvement in validation loss.")
                break

def save_model(model, path='mnist_cnn.pth'):
    print(f"Step 3: Saving model to {path}...")
    torch.save(model.state_dict(), path)
    if os.path.exists(path):
        print(f"Model saved successfully to {path}")
    else:
        print("Model saving failed!")

if __name__ == '__main__':
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    
    train_loader, val_loader = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting the training process...")
    train(model, train_loader, val_loader, criterion, optimizer, device)
    print("Training completed.")
    
    print("Saving the final model...")
    save_model(model)
