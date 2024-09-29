# Real-Time Object Detection with OpenCV and SSD MobileNet

This project implements real-time object detection using a webcam, the OpenCV library, and the SSD MobileNet model. It can detect various objects in the scene and draw bounding boxes around them, providing a visual interface to demonstrate the capabilities of machine learning in image processing.

## Features

- Real-time video feed from the webcam.
- Object detection using SSD MobileNet model.
- Bounding boxes are drawn around detected objects.
- Detected objects are labeled with their names and confidence scores.
- Frames per second (FPS) display for performance monitoring.

## Getting Started

### Prerequisites

1. **Python**: Make sure you have Python installed (Python 3.6 or later).
2. **OpenCV**: Install OpenCV library.
3. **NumPy**: Install NumPy for numerical operations.
4. **Coco Names File**: Download the `coco.names` file, which contains the names of the objects that can be detected.

### Install Dependencies

You can install the required packages using pip. Run the following command in your terminal:

```bash
pip install opencv-python numpy