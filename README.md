project-https:/[writing_with_fingers_webcam/](https://teja-0504.github.io/writing_with_fingers_webcam/templates/)                                                                                    
activation link-https://writing-with-fingers-webcam-5.onrender.com

original code in finger paint webcam.py
# Finger Paint Webcam

A real-time finger painting application using computer vision and hand tracking. Draw in the air using your fingers and see your creations on screen!

## Features

- **Real-time hand tracking** using MediaPipe
- **Finger gesture controls**:
  - 1 finger: Draw
  - 2 fingers: Move cursor
  - 3 fingers: Change color
  - 4 fingers: Clear canvas
  - 5 fingers: Undo last line
- **Web interface** with Flask backend
- **Multiple colors** (Yellow, Red, Green, Blue)
- **Live video stream** with drawing overlay

## Technologies Used

- **Python 3.10**
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand tracking and gesture recognition
- **Flask** - Web framework for API and streaming
- **NumPy** - Numerical operations
- **HTML/CSS/JavaScript** - Frontend interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/teja-0504/writing_with_fingers_webcam.git
cd writing_with_fingers_webcam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python webcam.py
```

4. Open your browser and go to:
```
http://localhost:5000
```

## Usage

### Gesture Controls

- **1 Finger Up**: Drawing mode - move your index finger to draw
- **2 Fingers Up**: Moving mode - move cursor without drawing
- **3 Fingers Up**: Change color (cycles through Yellow → Red → Green → Blue)
- **4 Fingers Up**: Clear entire canvas
- **5 Fingers Up**: Erase last drawn line

### Web Interface

- **Clear Button**: Clear the entire canvas
- **Change Color Button**: Cycle through available colors
- **Quit Button**: Stop the application

### Keyboard Shortcuts

- **'c'**: Clear canvas
- **'q'**: Quit application

## Live Demo

The application is deployed on Render: [https://writing-with-fingers-webcam-5.onrender.com/](https://writing-with-fingers-webcam-5.onrender.com/)

*Note: The live demo shows the interface but camera functionality requires local setup.*

## File Structure

```
writing_with_fingers_webcam/
├── webcam.py          # Main application with Flask backend
├── index.html         # Frontend web interface
├── requirements.txt   # Python dependencies
├── runtime.txt        # Python version for deployment
└── README.md         # This file
```

## Requirements

- Python 3.7-3.10
- Webcam/Camera access
- Modern web browser

## Development

To run in development mode:

1. Ensure your camera is accessible
2. Run `python webcam.py`
3. The Flask server will start on `http://localhost:5000`
4. Open the web interface in your browser

## Deployment

The app is configured for deployment on Render with:
- Python 3.10 runtime
- Automatic dependency installation
- MJPEG video streaming

## Troubleshooting

- **Camera not detected**: Ensure your camera is connected and not used by other applications
- **Permission denied**: Grant camera access to your browser/application
- **Module not found**: Install dependencies with `pip install -r requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
