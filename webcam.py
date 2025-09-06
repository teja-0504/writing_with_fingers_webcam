import cv2
import mediapipe as mp
import numpy as np
import threading
import flask
from flask import Flask, Response, request

class FingerPaint:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.canvas = None
        self.prev_x, self.prev_y = None, None

        self.colors = [(0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        self.color_index = 0

        self.lines = []

        self.total_fingers = 0
        self.running = False
        self.frame = None
        self.mode = "Idle"

    def fingers_up(self, hand_landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []
        # Thumb
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
        # Fingers
        for tip_id in tips_ids[1:]:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    fingers = self.fingers_up(hand_landmarks)
                    self.total_fingers = fingers.count(True)

                    h, w, c = frame.shape
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    cv2.putText(frame, f'Fingers: {self.total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (255, 0, 0), 3)

                    if self.total_fingers == 1:
                        self.mode = "Drawing"
                        if self.prev_x is not None and self.prev_y is not None:
                            line = ((self.prev_x, self.prev_y), (x, y), self.colors[self.color_index])
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.colors[self.color_index], 5)
                            self.lines.append(line)
                        self.prev_x, self.prev_y = x, y
                    elif self.total_fingers == 2:
                        self.mode = "Moving"
                        self.prev_x, self.prev_y = x, y
                    elif self.total_fingers == 3:
                        self.mode = "Color Change"
                        self.color_index = (self.color_index + 1) % len(self.colors)
                        cv2.putText(frame, f'Color Changed', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, self.colors[self.color_index], 3)
                        self.prev_x, self.prev_y = None, None
                    elif self.total_fingers == 5:
                        self.mode = "Erasing Line"
                        if self.lines:
                            self.lines.pop()
                            self.canvas = np.zeros_like(frame)
                            for line in self.lines:
                                cv2.line(self.canvas, line[0], line[1], line[2], 5)
                        self.prev_x, self.prev_y = None, None
                    elif self.total_fingers == 4:
                        self.mode = "Clear Canvas"
                        self.canvas = np.zeros_like(frame)
                        self.lines.clear()
                        self.prev_x, self.prev_y = None, None
                    else:
                        self.mode = "Idle"
                        self.prev_x, self.prev_y = None, None
            else:
                self.mode = "Idle"
                self.prev_x, self.prev_y = None, None

            combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
            cv2.putText(combined, f'Mode: {self.mode}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

            self.frame = combined
            cv2.imshow("Finger Paint - Press 'c' to clear, 'q' to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.canvas = np.zeros_like(frame)
                self.lines.clear()
            elif key == ord('q'):
                self.stop()
                break

app = Flask(__name__)
fp = None

def gen_frames():
    global fp
    while fp and fp.running:
        if fp.frame is not None:
            ret, buffer = cv2.imencode('.jpg', fp.frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            continue

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mode')
def get_mode():
    global fp
    return flask.jsonify({'mode': fp.mode if fp else 'Idle'})

@app.route('/clear', methods=['POST'])
def clear_canvas():
    global fp
    if fp:
        fp.canvas = np.zeros_like(fp.frame)
        fp.lines.clear()
    return '', 204

@app.route('/color', methods=['POST'])
def change_color():
    global fp
    if fp:
        fp.color_index = (fp.color_index + 1) % len(fp.colors)
    return '', 204

@app.route('/quit', methods=['POST'])
def quit_app():
    global fp
    if fp:
        fp.stop()
    return '', 204

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    fp = FingerPaint()
    fp.start()
    run_flask()
