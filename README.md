# OBJECT_DISTANCE_ESTIMATION

This project estimates the **distance between a camera and an object in an image**, using a custom-trained **YOLOv8  MiDas and a supervised learning ** model and real-world data collected on own.

---

## 🚀 Overview

The goal of this project is to:

- Detect a specific object in an image using **YOLOv8**
- Estimate the **distance from the camera to the object** using features extracted from the image.
- Support inference on custom images

---

## 📦 Dataset

Our Dataset is collected on our own using our own camera and pictures of different types of objects and different size objects were taken into consideration. 

📥 **Download Dataset**: [Click here to download](https://drive.google.com/file/d/11cRSx3T0gCUJ1_3SQeKxHfu97XcYwZVw/view?usp=sharing)

---

## 🧠 Model Details

- **Model Used**: [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Framework**: PyTorch (via Ultralytics YOLOv8)
- **Training**:
  - Used yolov8 for object detection and extraced bounding box features.
  - Trained on images of the object taken at different known distances using Random Forest.

---

## 📷 How It Works

1. You take a photo of the object using a monocular camera.
2. YOLOv8 detects the object and returns the bounding box features.
3. MiDas model used to extarct depth features.
4. Using bounding box and depth features created a own dataset.
5. Trained a Random Forest model using the data updated from the images.

---

## 📁 Project Structure

OBJECT_DISTANCE_ESTIMATION/
├── data/ # (Excluded from repo; download separately)
├── models/ # YOLOv8 weights
├── utils/ # Utility scripts
├── main.py # Main script for running inference
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🛠️ Setup Instructions

1. **Clone the repository**

```
git clone https://github.com/your-username/OBJECT_DISTANCE_ESTIMATION.git
cd OBJECT_DISTANCE_ESTIMATION
Create a virtual environment (optional but recommended)
```

```python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
```

```pip install -r requirements.txt
```

Download the dataset and other data requiremets here using this drive link :


Extract and place the data/ folder in the root directory.

🧪 Running Inference
python main.py --image path/to/image.jpg
Output:
Object detected at approximately 1.24 meters from the camera.
📈 Distance Estimation Logic
Uses a known real-world width of the object

Calculates focal length during calibration

Applies the formula:

ini
Copy code
distance = (real_width * focal_length) / perceived_width
📚 Dependencies
Python 3.8+

Ultralytics YOLOv8

OpenCV

NumPy

Install via:

pip install -r requirements.txt
📌 Notes
The accuracy depends on camera calibration and consistent lighting.

Works best with the same camera used during training.

You can improve results by increasing training data diversity.

🤝 Contributions
Contributions, pull requests, and suggestions are welcome!

📜 License
This project is licensed under the MIT License.

🧑‍💻 Author
Developed by Your Name
Reach me at: [your.email@example.com] or open an issue for support
