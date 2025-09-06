from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import time
import flask

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(0)

# Create a black canvas for drawing.
canvas = None

# Previous finger tip position.
prev_x, prev_y = None, None

# Color index for drawing
color_index = 0
colors = [(0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

# List to store drawn lines for undo
lines = []

# Global mode for keyboard controls
mode = 'none'
current_app_mode = "Idle"

def fingers_up(hand_landmarks, handedness):
    """
    Returns a list of booleans indicating which fingers are up.
    """
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: Compare tip and IP joint in x axis based on handedness.
    # Adjusted for right hand palm facing camera
    if handedness == 'Right':
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
    else:  # Left
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(True)
        else:
            fingers.append(False)

    # Fingers: tip higher than pip joint in y axis means finger is up.
    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

def generate_frames():
    global canvas, prev_x, prev_y, color_index, lines, mode, current_app_mode
    while True:
        ret, frame = cap.read()
        if not ret:
            # Create fallback frame when no camera is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Available", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            if canvas is None:
                canvas = np.zeros_like(frame)
            combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
            current_app_mode = "No Camera"
        else:
            # Flip the frame horizontally for natural interaction.
            frame = cv2.flip(frame, 1)

            # Initialize canvas if None.
            if canvas is None:
                canvas = np.zeros_like(frame)

            # Convert the frame color to RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and find hands.
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    # Draw hand landmarks on the frame.
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness
                    handedness = result.multi_handedness[i].classification[0].label

                    # Get finger states.
                    fingers = fingers_up(hand_landmarks, handedness)
                    total_fingers = fingers.count(True)

                    h, w, c = frame.shape
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    # Display finger count on frame
                    cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (255, 0, 0), 3)

                    # Override mode if keyboard mode is set
                    if mode != 'none':
                        current_mode = mode
                    else:
                        if total_fingers == 2:  # index + thumb
                            current_mode = 'write'
                        elif total_fingers == 1:  # index
                            current_mode = 'move'
                        elif total_fingers == 3:
                            current_mode = 'change_color'
                        elif total_fingers == 4:
                            current_mode = 'clear_all'
                        elif total_fingers == 5:
                            current_mode = 'remove_previous'
                        else:
                            current_mode = 'none'

                    if current_mode == 'write':
                        current_app_mode = "Drawing"
                        if prev_x is not None and prev_y is not None:
                            line = ((prev_x, prev_y), (x, y), colors[color_index])
                            cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], 5)
                            lines.append(line)
                        prev_x, prev_y = x, y
                    elif current_mode == 'move':
                        current_app_mode = "Moving"
                        prev_x, prev_y = x, y
                    elif current_mode == 'change_color':
                        current_app_mode = "Color Change"
                        color_index = (color_index + 1) % len(colors)
                        cv2.putText(frame, f'Color Changed', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, colors[color_index], 3)
                        prev_x, prev_y = None, None
                        if mode == 'change_color':
                            mode = 'none'
                    elif current_mode == 'remove_previous':
                        current_app_mode = "Erasing Line"
                        if lines:
                            lines.pop()
                            canvas = np.zeros_like(frame)
                            for line in lines:
                                cv2.line(canvas, line[0], line[1], line[2], 5)
                            time.sleep(0.5)  # give time
                        prev_x, prev_y = None, None
                        if mode == 'remove_previous':
                            mode = 'none'
                    elif current_mode == 'clear_all':
                        current_app_mode = "Clear Canvas"
                        canvas = np.zeros_like(frame)
                        lines.clear()
                        prev_x, prev_y = None, None
                        if mode == 'clear_all':
                            mode = 'none'
                    else:
                        current_app_mode = "Idle"
                        prev_x, prev_y = None, None
            else:
                current_app_mode = "Idle"
                prev_x, prev_y = None, None

            # Combine the frame and canvas.
            combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
            cv2.putText(combined, f'Mode: {current_app_mode}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', combined)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mode')
def get_mode():
    return flask.jsonify({'mode': current_app_mode})

@app.route('/clear', methods=['POST'])
def clear_canvas():
    global canvas, lines
    if canvas is not None:
        canvas = np.zeros_like(canvas)
        lines.clear()
    return '', 204

@app.route('/color', methods=['POST'])
def change_color():
    global color_index
    color_index = (color_index + 1) % len(colors)
    return '', 204

@app.route('/quit', methods=['POST'])
def quit_app():
    global cap
    cap.release()
    return '', 204

@app.route('/health')
def health_check():
    return flask.jsonify({'status': 'Backend running', 'mode': current_app_mode})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global mode
    data = request.get_json()
    mode = data.get('mode', 'none')
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
