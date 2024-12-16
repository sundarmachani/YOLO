# YOLO (You Only Look Once) Object Detection Implementation

## Project Overview

This project implements a miniaturized version of the YOLO (You Only Look Once) object detection system. YOLO is a state-of-the-art, real-time object detection algorithm that reframes object detection as a single regression problem, enabling end-to-end training and real-time processing speeds.

## Features

- Simplified YOLO architecture for CPU-based training and inference
- Synthetic dataset generation for rapid prototyping
- Custom loss function implementation
- Visualization tools for model predictions and training progress
- Efficient grid-based object detection

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mini-yolo.git
   cd mini-yolo
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model and visualize results:

```python
# View python notebook
YOLO.ipynb
```

This script will:
1. Generate a synthetic dataset
2. Train the YOLO model
3. Visualize training progress
4. Display model predictions on sample images

## Model Architecture

The miniaturized YOLO model consists of:

- Feature Extraction Backbone:
  - 3 convolutional blocks
  - Input: 224x224x3 RGB image
  - Output: 28x28x256 feature map

- Detection System:
  - Classification Head
  - Bounding Box Head
  - Grid-based prediction (7x7 grid)

## Training

- Batch Size: 16
- Learning Rate: 1e-4 (with step decay)
- Epochs: 20
- Loss Components:
  - Coordinate loss (λ=5.0)
  - Object confidence loss
  - No-object confidence loss (λ=0.5)
  - Class prediction loss

## Results

- Training Time: ~15 minutes (CPU)
- Final Loss: 439.75 (83.9% reduction from initial)
- Object Localization Accuracy: 98% (on synthetic data)
- Class Prediction Accuracy: 95%
- Inference Speed: 0.5s/image

## Limitations and Future Work

- CPU-only implementation limits scalability
- Synthetic dataset may not represent real-world complexity
- Single-scale detection
- Potential improvements:
  - Multi-scale feature detection
  - GPU acceleration
  - Integration with real datasets
