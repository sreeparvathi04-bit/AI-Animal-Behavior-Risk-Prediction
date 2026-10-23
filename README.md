# AI-Based Animal Behaviour and Risk Prediction System

## Project Overview
This project detects humans and animals in videos and predicts their behaviour and associated risk levels. It uses computer vision and machine learning to support animal behaviour monitoring.

## Classes
- Human
- Dog
- Elephant
- Monkey

## Behaviour Categories
- Calm
- Alert
- Stressed
- Aggressive

## Risk Levels
- Low
- Medium
- High

## Technologies Used
- Python
- YOLOv8 for object detection
- ByteTrack for object tracking
- Random Forest for behaviour and risk classification
- OpenCV for video processing
- Streamlit for the dashboard

## Main Features
- Video input and processing
- Human and animal detection
- Object tracking with track IDs
- Behaviour and risk prediction
- Risk warning messages
- Results and behaviour summaries

## Project Structure
- `app.py` — Streamlit dashboard
- `models/` — Model and tracking configuration files
- `test_detection.py` — Detection testing
- `test_model.py` — Model testing

## Installation
Install the required Python packages:
pip install streamlit ultralytics opencv-python pandas numpy joblib scikit-learn


## Run the Dashboard
From the project folder, run:
streamlit run app.py


## Note
The trained YOLO model (`best.pt`), datasets, and generated output videos are not included in this GitHub upload. The required model files must be available locally to run the complete system.

## Author
Sreeparvathi Dinesh

## Academic Project
Computer Science  
Sacred Heart College, Thevara

