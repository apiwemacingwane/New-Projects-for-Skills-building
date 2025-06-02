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


# Voice Recognition Chatbot

A Python-based voice-enabled chatbot that can listen to speech, process natural language, and respond with synthesized voice. Perfect for hands-free interaction and accessibility applications.

## 🎤 Features

- **Real-time Speech Recognition**: Converts your speech to text using Google's speech recognition API
- **Text-to-Speech Response**: Responds with natural-sounding synthesized voice
- **Intelligent Conversation**: Handles various topics including greetings, time/date queries, jokes, and general chat
- **Error Handling**: Gracefully manages unclear speech, timeouts, and connection issues
- **Conversation History**: Tracks and displays your entire chat session
- **Customizable Voice**: Adjustable speech rate, volume, and voice selection
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

- Python 3.6 or higher
- Working microphone
- Internet connection (for speech recognition)
- Speakers or headphones

### Installation

1. **Clone or download the script**
   ```bash
   # Save the voice_chatbot.py file to your local directory
   ```

2. **Install required packages**
   ```bash
   pip install SpeechRecognition pyttsx3 pyaudio
   ```

3. **Platform-specific setup** (if needed):

   **Windows:**
   ```bash
   # Usually works out of the box
   pip install pipwin
   pipwin install pyaudio
   ```

   **macOS:**
   ```bash
   # Install portaudio if you encounter issues
   brew install portaudio
   pip install pyaudio
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   # Install additional audio libraries
   sudo apt-get update
   sudo apt-get install python3-pyaudio
   sudo apt-get install espeak espeak-data libespeak-dev ffmpeg
   ```

### Running the Chatbot

```bash
python voice_chatbot.py
```

## 🎯 Usage

1. **Start the program** - Run the script and wait for microphone calibration
2. **Listen for the prompt** - The bot will say "Hello! I'm your voice assistant..."
3. **Start speaking** - Talk naturally when you see "Listening... (speak now)"
4. **Wait for response** - The bot will process your speech and respond with voice
5. **Continue conversation** - Keep talking naturally
6. **End conversation** - Say "goodbye", "bye", "quit", or "stop" to exit

### Example Conversation

```
Bot: Hello! I'm your voice assistant. You can talk to me naturally.
You: "Hello, how are you?"
Bot: I'm doing great, thank you for asking! How are you?
You: "What time is it?"
Bot: The current time is 2:30 PM.
You: "Tell me a joke"
Bot: Why don't scientists trust atoms? Because they make up everything!
You: "Goodbye"
Bot: Goodbye! It was nice talking with you.
```

## 🛠️ Configuration

### Voice Settings

Modify the `setup_tts()` method to customize voice properties:

```python
# Speech rate (words per minute)
self.tts_engine.setProperty('rate', 180)  # Default: 180

# Volume level (0.0 to 1.0)
self.tts_engine.setProperty('volume', 0.9)  # Default: 0.9

# Voice selection (if multiple voices available)
voices = self.tts_engine.getProperty('voices')
self.tts_engine.setProperty('voice', voices[1].id)  # Female voice
```

### Speech Recognition Settings

Adjust listening parameters in the `listen()` method:

```python
# Timeout for listening (seconds)
audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

# Microphone sensitivity
self.recognizer.adjust_for_ambient_noise(source)
```

## 🎨 Customization

### Adding New Response Patterns

Add custom responses in the `generate_response()` method:

```python
# Custom topic handling
if "weather" in user_input:
    return "I don't have access to real-time weather data, but I hope it's nice!"

# Add your own patterns
if "favorite color" in user_input:
    return "I like all colors, but blue is quite calming!"
```

### Extending Functionality

```python
# Add external API integration
def get_weather(self, location):
    # Integrate with weather API
    pass

# Add wake word detection
def detect_wake_word(self, audio):
    # Implement wake word detection
    pass
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pyaudio'`
**Solution**: 
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install python3-pyaudio
```

**Issue**: "Speech recognition error" or poor recognition accuracy
**Solutions**:
- Ensure stable internet connection
- Speak clearly and at normal pace
- Reduce background noise
- Adjust microphone sensitivity
- Move closer to microphone

**Issue**: No audio output or robotic voice
**Solutions**:
- Check system audio settings
- Install additional TTS voices
- Update audio drivers
- Try different voice selection in code

**Issue**: Microphone not detected
**Solutions**:
- Check microphone permissions
- Test microphone in other applications
- Use `sr.Microphone.list_microphone_names()` to debug

### Debug Mode

Add debugging information:

```python
# List available microphones
import speech_recognition as sr
print(sr.Microphone.list_microphone_names())

# List available voices
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}, ID: {voice.id}")
```

## 📋 Supported Commands

The chatbot recognizes various types of input:

- **Greetings**: "hello", "hi", "hey", "greetings"
- **Farewells**: "goodbye", "bye", "quit", "stop", "exit"
- **Time queries**: "what time is it", "current time"
- **Date queries**: "what's the date", "today's date"
- **Identity**: "what's your name", "who are you"
- **Health check**: "how are you"
- **Help**: "help me", "what can you do"
- **Entertainment**: "tell me a joke", "something funny"
- **General conversation**: Open-ended questions and statements

## 🔒 Privacy & Security

- Speech is processed using Google's speech recognition service
- No conversation data is stored permanently
- Local text-to-speech processing
- No personal data is transmitted beyond speech recognition

## 🚧 Limitations

- Requires internet connection for speech recognition
- Recognition accuracy depends on audio quality and accent
- Limited to English language (can be extended)
- Basic conversation logic (can be enhanced with AI/ML)
- No persistent memory between sessions

## 🔮 Future Enhancements

- **Offline speech recognition** using local models
- **AI/ML integration** for more intelligent responses
- **Multi-language support**
- **Wake word detection** for hands-free activation
- **External API integration** (weather, news, etc.)
- **Voice training** for better user recognition
- **Emotion detection** in speech
- **Context awareness** for longer conversations

## 🤝 Contributing

Feel free to contribute improvements:

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Test with the debug commands provided
- Ensure all dependencies are properly installed
- Verify microphone and speaker functionality

## 🎉 Acknowledgments

- Built with `SpeechRecognition` library
- Uses `pyttsx3` for text-to-speech
- Powered by Google Speech Recognition API
