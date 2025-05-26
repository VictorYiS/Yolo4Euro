# Pure Vision Autonomous Driving System

An AI-powered autonomous driving system for Euro Truck Simulator 2, featuring real-time lane detection, traffic light recognition, and intelligent vehicle control.

Video demo: [YouTube Link](https://www.youtube.com/watch?v=gsqwVLMu3XQ)

## Features

- **Lane Detection**: Real-time lane line detection using E-Net deep learning model
- **Traffic Recognition**: YOLO-based traffic light and vehicle detection
- **Autonomous Control**: PID-based steering and speed control system
- **Multi-Process Architecture**: Efficient parallel processing for real-time performance
- **Dashboard Interface**: Real-time display of vehicle telemetry data
- **Safety Systems**: Emergency stop and manual override capabilities

## System Architecture

The system consists of several key components:

- **Main Process** (`main.py`): Orchestrates the entire system and manages inter-process communication
- **Dashboard** (`dashboard.py`): Collects and processes vehicle telemetry data
- **Lane Detection** (`detector.py`): E-Net model for lane segmentation and YOLO for object detection
- **Control Handler** (`control_handler.py`): Manages autonomous driving logic and keyboard controls
- **Driver** (`driver.py`): PID controller for steering and acceleration decisions
- **Data Loader** (`data_loader.py`): Interfaces with truck telemetry system

## Requirements

### Hardware
- NVIDIA GPU (recommended for real-time inference)
- Minimum 8GB RAM
- Screen capture capability
- You need to own a compatible truck simulation game (e.g., Euro Truck Simulator 2)

### Software Dependencies
```bash
pip install torch torchvision # I use version 2.5.1+cu118
pip install ultralytics  # YOLO
pip install opencv-python
pip install numpy
pip install pillow
pip install mss  # Screen capture
pip install pynput  # Keyboard control
pip install pytesseract  # OCR for dashboard
pip install scikit-learn
pip install matplotlib
pip install tqdm
```

### Additional Requirements
- Tesseract OCR engine
- Compatible truck simulation game
- CUDA-capable GPU (optional but recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
   - Place E-Net model weights in `models/ENET.pth`
   - Place YOLO model weights in `models/1000m_736sgz.pt`
   - Put the `scs-telemetry.dll` file in `bin\win_x64\plugins\` of the game installation directory. If there is no plugins folder, you need to create it yourself.

4. Install Tesseract OCR:
   - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### Quick Start

1. Launch your truck simulation game
2. Run the autonomous driving system:
```bash
python main.py
```

### Control Keys

- **Z**: Toggle autonomous driving mode on/off
- **X**: Emergency stop (deactivates autonomous mode)
- **ESC**: Exit the program

### Configuration

Edit `window.py` to adjust:
- Game window resolution settings
- ROI (Region of Interest) parameters
- Coordinate scaling factors

### Model Configuration

The system uses two main AI models:

1. **E-Net Lane Detection**: 
   - Input: Grayscale images (512x256)
   - Output: Binary segmentation + Instance segmentation
   - Model path: `models/ENET.pth`
   - E-NET model and network code is used and learnt from here https://www.kaggle.com/code/rangalamahesh/lane-inference-test-enet-model/input

2. **YOLO Traffic Detection**:
   - Input: RGB images
   - Output: Bounding boxes with classifications
   - Model path: `models/1000m_736sgz.pt`, download path:https://drive.google.com/file/d/1g37xk1_V27iBMFgZpsB_jedDnTSkrDbA/view?usp=sharing

A in-game plugin `scs-telemetry.dll` is required to collect vehicle telemetry data. This plugin must be placed in the game's plugins directory.
Download path:https://drive.google.com/file/d/1x1YGcFTE-yE-Tx1IqijUL7ap3yEu9UOC/view?usp=sharing

## System Workflow

1. **Screen Capture**: Continuously captures game screen using MSS
2. **Data Processing**: Extracts vehicle telemetry and visual data
3. **AI Inference**: 
   - Lane detection using E-Net
   - Traffic/vehicle detection using YOLO
4. **Decision Making**: PID controller calculates steering and acceleration
5. **Action Execution**: Sends keyboard commands to control the vehicle

## Performance Optimization

- **Multi-processing**: Separate processes for data collection and control logic
- **Shared Memory**: Efficient data transfer between processes
- **Action Batching**: Intelligent key press batching for smoother control
- **GPU Acceleration**: CUDA support for faster inference

## Safety Features

- **Manual Override**: Instant manual control takeover
- **Emergency Stop**: Immediate braking capability
- **Stability Control**: Anti-oscillation steering algorithms
- **Speed Limiting**: Automatic speed adjustment based on road conditions

## Troubleshooting

### Common Issues

1. **Game Window Not Detected**:
   - Ensure the game is running and visible
   - Check `window.py` resolution settings
   - Verify template matching threshold

2. **Poor Lane Detection**:
   - Check lighting conditions in game
   - Adjust ROI parameters
   - Verify E-Net model path

3. **Delayed Response**:
   - Reduce inference resolution
   - Check GPU utilization
   - Optimize action queue processing

### Debug Features

- Set `open_log=True` in NumberWindow for OCR debugging
- Check `logs/running.log` for detailed system logs
- Use `debug_images/` folder for visual debugging

## Model Training

### E-Net Lane Detection

The E-Net model can be retrained using:
```python
from train_e_net import LaneDataset, ENet
# Training code available in train_e_net.py
```

### YOLO Traffic Detection

Use the Ultralytics YOLO framework for custom training:
```bash
yolo train data=custom_dataset.yaml model=yolo11n.pt epochs=100
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Performance Metrics

- **Frame Rate**: 30+ FPS on RTX 4070
- **Lane Detection Accuracy**: >95% on clear roads
- **Response Time**: <50ms from detection to action
- **Memory Usage**: ~2GB RAM typical operation

## License

This project is for educational and research purposes. Please ensure compliance with the terms of service of any games used with this system.

## Acknowledgments

- E-Net architecture for efficient semantic segmentation
- YOLO framework for real-time object detection
- TuSimple dataset for lane detection training
- Ultralytics for YOLO implementation

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs in `logs/running.log`
3. Open an issue on GitHub with detailed information

---

**⚠️ Warning**: This system is designed for simulation purposes only. Do not use with real vehicles. Always maintain manual oversight when using autonomous features.