# YOLO-Real-Time-Object-Detection-Webcam-

📌 Project Overview
This project implements real-time object detection using the YOLO (You Only Look Once) algorithm with a live webcam feed. It detects multiple objects in each video frame, draws bounding boxes around them, and displays the class label and confidence score in real-time.
The project demonstrates how YOLO can process frames fast enough for live applications, making it suitable for interactive use cases such as surveillance, robotics, and automation.

✨ Features
🖥️ Detects multiple objects simultaneously in a webcam feed.
🟦 Displays bounding boxes, class labels, and confidence scores in real-time.
⚡ Uses YOLOv3 / YOLOv3-tiny for fast detection.
🔧 Supports customizable detection thresholds.
💾 Handles large model weights using Git Large File Storage (LFS).

🛠️ Technologies Used
Python 3.11
OpenCV (DNN module)
NumPy
YOLOv3
COCO Dataset

📁 Project Structure
YOLO-Real-Time-Object-Detection/
│
├── cfg/
│ ├── yolov3.cfg
│ └── yolov3-tiny.cfg
│
├── weights/
│ ├── yolov3.weights
│ └── yolov3-tiny.weights
│
├── coco.names
├── real_time_yolo_webcam.py
├── real_time_yolo_detector1.py
├── real_time_yolo_detector2.py
├── README.md

🧠 How It Works
🎥 Capturing Frames:
The webcam continuously captures frames as input images for YOLO.
📐 Grid Division & Prediction:
YOLO divides each frame into a grid and predicts bounding boxes and class probabilities for each cell.
⚡ Single Forward Pass:
All predictions are generated in one pass, enabling real-time performance.
❌ Non-Maximum Suppression:
Overlapping boxes for the same object are filtered to retain only the most confident prediction.
🖌️ Visualization:
Detected objects are highlighted on the frame with bounding boxes, labels, and confidence scores.

📚 Dataset
COCO Dataset (Common Objects in Context)

💻 Installation

Clone the repository:
git clone https://github.com/Shikha18Shukla/YOLO-Real-Time-Object-Detection-Webcam-.git
cd YOLO-Real-Time-Object-Detection

Set up a virtual environment (optional but recommended):
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Download YOLO weights (if not included):
YOLOv3: ⬇️ Download link
YOLOv3-tiny: ⬇️ Download link

▶️ Usage
Run the main script to start real-time detection:
python detect.py
The webcam window will open, showing live detection.
Press q to quit the application.

📚 References
YOLO: You Only Look Once
Git LFS

