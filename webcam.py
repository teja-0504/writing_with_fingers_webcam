from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

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

# Define a dictionary for gesture-to-mode mapping
GESTURE_MODES = {
    2: 'write',
    1: 'move',
    3: 'change_color',
    4: 'clear_all',
    5: 'remove_previous'
}

def fingers_up(hand_landmarks, handedness):
    """
    Returns a list of booleans indicating which fingers are up.
    """
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: Compare tip and IP joint in x axis based on handedness.
    if handedness == 'Right':
        fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
    else:  # Left
        fingers.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)

    # Fingers: tip higher than pip joint in y axis means finger is up.
    for tip_id in tips_ids[1:]:
        fingers.append(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y)
    return fingers

def generate_frames():
    global canvas, prev_x, prev_y, color_index, lines, mode
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Start with a default mode of 'none'
        current_mode = mode if mode != 'none' else 'none'

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handedness = result.multi_handedness[i].classification[0].label
                fingers = fingers_up(hand_landmarks, handedness)
                total_fingers = fingers.count(True)
                
                # Get the mode from the dictionary
                current_mode = GESTURE_MODES.get(total_fingers, 'none')

                h, w, c = frame.shape
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 0, 0), 3)

                if current_mode == 'write':
                    if prev_x is not None and prev_y is not None:
                        line = ((prev_x, prev_y), (x, y), colors[color_index])
                        cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], 5)
                        lines.append(line)
                    prev_x, prev_y = x, y
                elif current_mode == 'change_color':
                    color_index = (color_index + 1) % len(colors)
                    cv2.putText(frame, f'Color Changed', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, colors[color_index], 3)
                    prev_x, prev_y = None, None
                elif current_mode == 'remove_previous':
                    if lines:
                        lines.pop()
                        canvas = np.zeros_like(frame)
                        for line in lines:
                            cv2.line(canvas, line[0], line[1], line[2], 5)
                        time.sleep(0.5)
                    prev_x, prev_y = None, None
                elif current_mode == 'clear_all':
                    canvas = np.zeros_like(frame)
                    lines.clear()
                    prev_x, prev_y = None, None
                else: # Handles 'move' and 'none' cases
                    if current_mode == 'move':
                        prev_x, prev_y = x, y
                    else:
                        prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Reset keyboard mode after it has been used
        if mode != 'none' and current_mode == mode:
            mode = 'none'

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global mode
    data = request.get_json()
    mode = data.get('mode', 'none')
    return 'OK'

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global cap
    cap.release()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(debug=True)
