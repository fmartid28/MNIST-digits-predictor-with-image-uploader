from model import written_numbers_CNN  # Import your architecture
from PIL import Image
from torchvision import transforms

# 1. Load the Model
model = written_numbers_CNN()
model.load_state_dict(torch.load('../models/mnist_pytorch.pth', weights_only=True))
model.eval()

# 2. Define the Preprocessing (matching the training input and normalization)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension [1, 1, 28, 28]

    with torch.no_grad(): # Disable gradient calculation (faster/saves memory)
        output = model(img_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
        
    return prediction.item()

# Usage
print(f"The model thinks this is a: {predict_digit('')}")