from flask import Flask, render_template, request
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_map = {
    0: "CNV",
    1: "DME",
    2: "DRUSEN",
    3: "NORMAL"
}

class OctMNISTModel(nn.Module):
    def __init__(self, num_classes=4):
        super(OctMNISTModel, self).__init__()
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv2 = nn.Dropout(p=0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # Adjusted after calculation
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # First convolution and pooling
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Second convolution and pooling
        x = x.view(x.size(0), -1)  # Flattening
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = OctMNISTModel()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    result = ''
    file_path = ''
    file_url = ''

    file = request.files["file"]

    if file.filename == "":
        return render_template('index.html')

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        image_tensor = preprocess_image(file_path)

        print("Shape of image -")
        print(image_tensor.shape)

        with torch.no_grad():
            output = model(image_tensor)

        result = torch.argmax(output, dim=1).item()

        result = class_map[result]

        file_url = "https://test-deploy-531626027781.us-east1.run.app/static/uploads/" + file.filename

    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
