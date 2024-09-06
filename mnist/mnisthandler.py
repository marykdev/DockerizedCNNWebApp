from ts.torch_handler.base_handler import BaseHandler
import torch
from PIL import Image
from torchvision import transforms

class MNISTHandler(BaseHandler):
    def __init__(self):
        super(MNISTHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(context)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        image = Image.open(data[0]['body']).convert('L')  # Convert image to grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # MNIST image size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(image).unsqueeze(0)

    def inference(self, data, *args, **kwargs):
        if not self.initialized:
            raise Exception('Model is not initialized.')
        with torch.no_grad():
            image = data[0]
            output = self.model(image.to(self.device))
            _, predicted = torch.max(output, 1)
            return predicted.cpu().numpy().tolist()

    def postprocess(self, data):
        return data
