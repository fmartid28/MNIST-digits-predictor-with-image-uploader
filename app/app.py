import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Add 'src' to path so we can import the model architecture
sys.path.append(os.path.abspath("./src"))
from model import written_numbers_CNN

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="Digit Convolutional Neural Net FM", layout="centered")

@st.cache_resource
def load_digit_model():
    # Initialize architecture
    model = written_numbers_CNN()
    # Load weights (mapping to CPU for compatibility)
    model.load_state_dict(
        torch.load('models/mnist_pytorch.pth', map_location='cpu', weights_only=True)
    )
    model.eval()
    return model

model = load_digit_model()

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- USER INTERFACE ---
st.title("Handwritten Digit Classifier")
st.markdown("Upload a clear black-background, white-font image of a single digit (0-9).")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process and Predict
    with col2:
        with st.spinner('Analyzing...'):
            # Prepare tensor
            img_tensor = transform(image).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                prediction = output.argmax(dim=1).item()
                # Optional: Get confidence (softmax)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence = torch.max(prob).item() * 100

            st.metric(label="Predicted Digit", value=prediction)
            st.write(f"**Confidence:** {confidence:.2f}%")

else:
    st.info("Please upload an image file in the sidebar to begin.")