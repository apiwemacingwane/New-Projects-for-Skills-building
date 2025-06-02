# Hand Tracking Drag-and-Drop System

A computer vision application that enables touchless drag-and-drop interaction using hand gestures. Control objects on screen by pinching your fingers together - no mouse or touch required!

![Demo](https://img.shields.io/badge/Status-Ready%20to%20Use-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-orange)

## 🎯 Features

- **Touchless Interaction**: Control objects using hand gestures captured by your webcam
- **Pinch-to-Grab**: Bring index and middle fingers together to grab objects
- **Real-time Tracking**: Smooth, responsive hand tracking with visual feedback
- **Multiple Objects**: Support for multiple draggable items simultaneously
- **Auto-Generated Samples**: Creates colorful sample shapes if no images are provided
- **Image Support**: Load your own PNG, JPG, or JPEG files as draggable objects
- **Visual Feedback**: Clear indicators showing grab state and finger positions
- **Reset Functionality**: Press 'R' to reset all object positions

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed, then install the required packages:

```bash
pip install opencv-python cvzone numpy
```

### Running the Application

1. **Clone or download** the script to your computer
2. **Run the script**:
   ```bash
   python hand_tracking_drag_drop.py
   ```
3. **Start interacting**:
   - Position yourself in front of your webcam
   - Hold up one hand in view of the camera
   - Pinch your index and middle fingers together over an object to grab it
   - Move your hand while pinching to drag the object
   - Release the pinch to drop the object

## 🎮 Controls

| Gesture/Key | Action |
|-------------|--------|
| **Pinch Fingers** | Grab/Release objects (bring index & middle fingers together) |
| **Move While Pinching** | Drag the grabbed object |
| **R Key** | Reset all object positions |
| **ESC Key** | Exit the application |

## 📁 Using Your Own Images

### Option 1: Use Sample Shapes (Default)
- Simply run the application without any setup
- Five colorful sample shapes will be automatically created:
  - Red Circle
  - Green Square
  - Blue Triangle
  - Yellow Diamond
  - Magenta Heart

### Option 2: Load Your Own Images
1. Place your image files (PNG, JPG, JPEG) in the same folder as the script
2. Run the application - your images will be loaded automatically
3. Images larger than 200px will be automatically resized for optimal performance

**Supported formats**: PNG, JPG, JPEG
**Recommended size**: 50-200 pixels for best performance

## 🛠️ Technical Details

### How It Works

1. **Hand Detection**: Uses MediaPipe (via CVZone) to detect hand landmarks in real-time
2. **Gesture Recognition**: Monitors distance between index and middle fingertips
3. **Grab Detection**: When fingers are close enough (< 40 pixels), a "pinch" is detected
4. **Object Interaction**: Checks if pinch occurs over a draggable object
5. **Visual Rendering**: Overlays transparent images onto the camera feed

### Key Components

- **Hand Tracking**: CVZone HandDetector with optimized settings
- **Image Processing**: OpenCV for camera capture and image manipulation
- **Transparency Handling**: Custom alpha blending for PNG overlays
- **Collision Detection**: Point-in-rectangle collision for grab detection

### Performance Optimizations

- Single hand tracking for better performance
- Efficient alpha blending algorithm
- Automatic image resizing for large files
- Optimized drawing with minimal frame processing

## 🎨 Customization

### Adjusting Sensitivity
Modify the grab sensitivity by changing this value in the code:
```python
GRAB_DISTANCE_THRESHOLD = 40  # Lower = more sensitive, Higher = less sensitive
```

### Changing Initial Positions
Customize where objects appear initially:
```python
start_x, start_y = 100, 100        # Starting position
offset_between = 200               # Spacing between objects
```

### Camera Settings
Adjust camera resolution:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
```

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

**Hand tracking not working:**
- Ensure good lighting conditions
- Keep your hand clearly visible in the camera frame
- Try adjusting the detection confidence: `detectionCon=0.7` (lower = more sensitive)

**Performance issues:**
- Close other applications using the camera
- Reduce camera resolution in the code
- Ensure adequate lighting to help hand detection

**No objects appear:**
- Check that image files are in the correct directory
- Verify image files are in supported formats (PNG, JPG, JPEG)
- If using sample shapes, they should appear automatically

### Warning Messages

You may see TensorFlow/MediaPipe warnings when starting - these are normal and don't affect functionality:
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called...
```

## 📋 System Requirements

- **Python**: 3.7 or higher
- **Webcam**: Any USB or built-in camera
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB recommended
- **Processor**: Modern multi-core processor recommended for smooth performance

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

### Ideas for Enhancement
- Multi-hand support
- Gesture-based object rotation
- Object snapping/alignment
- Save/load object positions
- Custom gesture recognition
- 3D hand tracking

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **CVZone**: For the excellent hand tracking wrapper
- **OpenCV**: For computer vision capabilities
- **MediaPipe**: For the underlying hand detection model
- **NumPy**: For efficient array operations

---

**Enjoy your touchless drag-and-drop experience!** 🚀

For questions or support, please create an issue in the repository.
