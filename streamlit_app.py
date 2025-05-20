import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(
    page_title="PigDoc - AI-Powered Pig Disease Detection",
    page_icon="🐷",
    layout="centered"
)

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Get model interpreter
interpreter = load_model()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define disease classes
DISEASE_CLASSES = [
    "Healthy",
    "Respiratory Disease",
    "Skin Disease",
    "Gastrointestinal Disease"
]

def preprocess_image(image):
    # Resize image to match model input shape
    input_shape = input_details[0]['shape']
    img = cv2.resize(np.array(image), (input_shape[1], input_shape[2]))
    
    # Convert to RGB if needed
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # If RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = float(output_data[0][prediction])
    
    return DISEASE_CLASSES[prediction], confidence

# UI Elements
st.title("🐷 PigDoc")
st.subheader("AI-Powered Pig Disease Detection System")

st.markdown("""
This application uses machine learning to detect potential health issues in pigs through image analysis.
Simply upload an image of a pig, and the system will analyze it for possible health conditions.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a spinner during prediction
    with st.spinner("Analyzing image..."):
        try:
            # Get prediction
            disease, confidence = predict_disease(image)
            
            # Display results
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Condition", disease)
            with col2:
                st.metric("Confidence", f"{confidence*100:.2f}%")
            
            # Additional information based on condition
            st.subheader("Recommendations")
            if disease == "Healthy":
                st.info("The pig appears to be in good health. Continue with regular care and monitoring.")
            else:
                st.warning(f"""
                Potential {disease} detected. Recommended actions:
                - Consult with a veterinarian
                - Isolate the affected animal if necessary
                - Monitor temperature and eating habits
                - Document symptoms and changes
                """)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            
st.markdown("---")
st.markdown("""
### Usage Tips
- Ensure the image is clear and well-lit
- The pig should be clearly visible in the image
- Multiple pigs in one image may affect accuracy
""")

# Add GitHub link
st.sidebar.markdown("### Project Information")
st.sidebar.markdown("[View on GitHub](https://github.com/KigothoKen/PigDoc)") 