import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import CNNModel

def load_test_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss /= total
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNModel().to(device)
    model.load_state_dict(torch.load('mnist_cnn_best.pth'))
    
    test_loader = load_test_data()
    criterion = torch.nn.CrossEntropyLoss()
    
    test(model, test_loader, criterion, device)
