import streamlit as st
import torch
import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import asyncio
import pandas as pd

# ✅ Fix Asyncio issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
yolo_model = YOLO("my_model/train/weights/best.pt")

# ✅ Load Random Forest Model Safely
rf_model_path = "random_forest_distance.pkl"

try:
    rf_model = joblib.load(rf_model_path)
    if not hasattr(rf_model, "predict"):
        raise ValueError("Loaded file is not a valid Random Forest model.")
except Exception as e:
    st.error(f"❌ Error loading Random Forest model: {e}")
    rf_model = None

# ✅ Load MiDaS model only when script runs
torch.hub.set_dir("/tmp/torch_cache") 
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()

# ✅ Define MiDaS transformation
midas_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def estimate_depth(image, bbox):
    """Extract mean and median depth from the bounding box."""
    x1, y1, x2, y2 = bbox

    # Convert to RGB and apply transformation
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transform(img_rgb).unsqueeze(0).to(device)

    # Run MiDaS model
    with torch.no_grad():
        depth_map = midas(input_tensor).squeeze(0).cpu().numpy()

    # Resize depth map to match original image size
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

    # Extract depth values within bounding box
    object_depth = depth_map[y1:y2, x1:x2]

    if object_depth.size == 0:
        return None, None

    median_depth = np.median(object_depth)
    mean_depth = np.mean(object_depth)

    return median_depth, mean_depth

def predict_distance(y_min, y_max, median_depth, mean_depth):
    """Predict distance using the trained Random Forest model."""
    if rf_model is None:
        st.warning("⚠ Random Forest model is NOT loaded!")
        return None
    if median_depth is None or mean_depth is None:
        st.warning("⚠ Depth values are missing!")
        return None

    # **🔥 Fixing Feature Order**
    features = np.array([[y_min,y_max,median_depth,mean_depth]])
    
    st.write(f"📊 Features for prediction: {features}")  # Debugging output
    
    predicted_distance = rf_model.predict(features)[0]
    st.write(f"🔮 Predicted Distance: {predicted_distance:.2f}m")  # Debugging output
    
    return float(predicted_distance)

# ✅ Streamlit UI
st.title("Object Distance Estimation")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    image = np.array(Image.open(uploaded_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO for object detection
    results = yolo_model(image)

    # Initialize result storage
    object_data = []

    # Process detections
    if results[0].boxes is not None:
        for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)

            # Estimate depth
            median_depth, mean_depth = estimate_depth(image, (x1, y1, x2, y2))

            # Predict distance from the loaded Random Forest model
            distance = predict_distance(y1, y2, median_depth, mean_depth)

            # Store data
            object_data.append({
                "Object #": idx + 1,
                "y_min": y1,
                "y_max": y2,
                "Median Depth": round(median_depth, 2) if median_depth else "N/A",
                "Mean Depth": round(mean_depth, 2) if mean_depth else "N/A",
                "Predicted Distance (m)": round(distance, 2) if distance else "N/A"
            })

            # Draw bounding box and text
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Dist: {distance:.2f}m" if distance else "Dist: N/A"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert image back to RGB for Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Processed Image", use_container_width=True)  # ✅ Fix `use_column_width` warning

    # Display detection results
    if object_data:
        df = pd.DataFrame(object_data)
        st.write("### Random Forest Predictions")
        st.table(df)
    else:
        st.write("### No objects detected.")

