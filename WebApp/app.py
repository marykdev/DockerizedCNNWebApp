from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Define transforms for each model
def get_transform(model_name):
    if model_name == 'mnist':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),  # MNIST image size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:  # Default to ChestXray
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

# Load the appropriate model
def load_model(model_name):
    model_paths = {
        'chestxray': 'Code/ChestXray/chestxray_best_model.pth',
        'mnist': 'Code/mnist/mnist_best_model.pth'
    }
    model_classes = {
        'chestxray': 'Code/ChestXray/chestxray_model.py',
        'mnist': 'Code/mnist/mnist_model.py'
    }
    
    if model_name not in model_paths:
        raise ValueError("Invalid model name")

    # Dynamically import the model class
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_classes[model_name])
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    if model_name == 'chestxray':
        model = model_module.ChestXrayCNN()
    elif model_name == 'mnist':
        model = model_module.CNNModel()
    
    model.load_state_dict(torch.load(model_paths[model_name]))
    model.eval()
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file)
    
    # Determine model based on image dimensions
    width, height = image.size
    if width == 28 and height == 28:
        model_name = 'mnist'
    elif width == 128 and height == 128:
        model_name = 'chestxray'
    else:
        return jsonify({'error': 'Unable to determine model based on image dimensions'})
    
    transform = get_transform(model_name)
    image = transform(image).unsqueeze(0)

    model = load_model(model_name)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
