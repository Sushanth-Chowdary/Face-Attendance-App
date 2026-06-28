# Face Attendance App

📸 Automated Attendance System using Face Recognition

This is a web application built with Streamlit and PyTorch that automates the attendance process by recognizing faces from images, videos, and live camera feeds.

## Features
- **Dashboard**: System overview and list of trained students.
- **Add New Student**: Upload photos to train the system for a new student.
- **Video Attendance**: Upload a class video to automatically mark attendance. Output is exportable as a CSV file.
- **Live Camera Attendance**: Use your webcam or upload a photo to identify faces and mark attendance on a single frame.

## Setup Instructions

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Directory Structure
- `app.py`: The Streamlit web interface
- `face_utils.py`: The backend logic for face detection and recognition using facenet-pytorch
- `labels/`: Directory where student training photos are stored
- `face_attendance_model.pkl`: The trained face recognition model (created after adding students)
- `attendance_sheet.csv`: Output log of attendance
