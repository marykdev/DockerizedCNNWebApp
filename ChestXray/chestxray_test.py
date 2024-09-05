import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from chestxray_model import ChestXrayCNN

def load_data(batch_size=64):
    print("Loading test data...")
    
    # Define the absolute paths
    base_path = os.path.abspath('./chestxray_data')
    test_path = os.path.join(base_path, 'test')

    # Print path for debugging
    print(f"Absolute Test Path: {test_path}")

    # Check if directory exists
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Test data directory not found: {test_path}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Create data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test data loaded. Size: {len(test_dataset)}")
    return test_loader

def test(model, test_loader, criterion, device):
    print("Testing model...")
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXrayCNN().to(device)
    
    # Load the best model
    model_path = 'chestxray_best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    test_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    
    print("Starting the testing process...")
    test(model, test_loader, criterion, device)
    print("Testing completed.")
