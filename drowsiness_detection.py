import sys
import cv2
import torch
import numpy as np
import threading
import pygame  # For alarm control
from transformers import pipeline
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# Initialize pygame mixer
pygame.mixer.init()

# Load alarm sound (Ensure this file exists)
alarm_sound = "alarm.mp3"
pygame.mixer.music.load(alarm_sound)

# Load Hugging Face model
pipe = pipeline("image-classification", model="chbh7051/driver-drowsiness-detection")

class DrowsinessDetector(QWidget):
    def __init__(self):
        super().__init__()

        # UI Setup
        self.initUI()

        # Webcam
        self.cap = cv2.VideoCapture(0)
        
        # Drowsiness tracking
        self.drowsy_detected = False
        self.detection_running = False  # Track if detection is active

        # Timer for real-time detection
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        """Initialize GUI layout and components"""
        self.setWindowTitle("Driver Drowsiness Detection")
        self.setGeometry(100, 100, 800, 600)

        # Video Feed Label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        # Status Label
        self.status_label = QLabel("Status: Click 'Start Detection'", self)

        # Start Button
        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.start_detection)

        # Stop Button
        self.stop_button = QPushButton("Stop Detection", self)
        self.stop_button.clicked.connect(self.stop_detection)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

    def start_detection(self):
        """Start real-time drowsiness detection"""
        if not self.detection_running:
            self.detection_running = True
            self.timer.start(30)

    def stop_detection(self):
        """Stop real-time drowsiness detection"""
        self.detection_running = False
        self.timer.stop()
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: black;")
        self.drowsy_detected = False  # Stop the alarm
        pygame.mixer.music.stop()  # Stop the alarm immediately

    def update_frame(self):
        """Capture frame, detect drowsiness, and update GUI"""
        if not self.detection_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert OpenCV frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.0, beta=0)  # Adjust brightness and contrast


        # Resize for model
        resized_frame = cv2.resize(rgb_frame, (224, 224))

        # Convert to PIL Image
        pil_image = Image.fromarray(resized_frame)

        # Run inference
        results = pipe(pil_image)
        label = results[0]['label'].lower()

        # Update status based on prediction
        if label == "drowsy":
            self.status_label.setText("Status: Drowsy! ⚠️")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

            # Start alarm only if not already playing
            if not self.drowsy_detected:
                self.drowsy_detected = True
                pygame.mixer.music.play(-1)  # Loop the alarm
        else:
            self.status_label.setText("Status: Awake 😊")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.drowsy_detected = False
            pygame.mixer.music.stop()  # Stop alarm immediately

        # Convert frame to PyQt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Cleanup when closing window"""
        self.cap.release()
        pygame.mixer.music.stop()
        event.accept()

# Run GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
