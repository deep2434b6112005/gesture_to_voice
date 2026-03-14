import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ==============================
# 1. Threaded Camera Class
# ==============================
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        self.cap.release()


# ==============================
# 2. Load Model
# ==============================
try:
    model = load_model("gesture_gru_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# ==============================
# 3. MediaPipe Setup
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==============================
# 4. Settings & Hyperparameters
# ==============================

# Faster prediction
SEQ_LENGTH = 20

sequence = deque(maxlen=SEQ_LENGTH)
sentence_buffer = []

# Faster smoothing
pred_history = deque(maxlen=15)

CONF_THRESHOLD = 0.85
STABILITY_FRAMES = 10

last_word = ""

# Cooldown to prevent repetition
last_prediction_time = 0
COOLDOWN_TIME = 1.2

frame_counter = 0

gesture_labels = {
    1: "Hungry",
    0: "I am",
    2: "want",
    3: "some water",
    4: "look",
    5: "beautiful",
    6: "help me",
    7: "how many",
    8: "to buy",
    9: "sorry"
}


# ==============================
# 5. Start Stream
# ==============================
vs = VideoStream("http://10.31.181.196:8080/video")


# ==============================
# 6. Main Loop
# ==============================
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        ret, frame = vs.read()
        if not ret:
            continue

        frame_counter += 1

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            res_hands = results.multi_hand_landmarks[0]
            wrist = res_hands.landmark[0]

            # ---- Feature Normalization ----
            dist = np.sqrt(
                (res_hands.landmark[9].x - wrist.x) ** 2 +
                (res_hands.landmark[9].y - wrist.y) ** 2
            )

            if dist == 0:
                dist = 1

            lm_list = []

            for lm in res_hands.landmark:
                lm_list.extend([
                    (lm.x - wrist.x) / dist,
                    (lm.y - wrist.y) / dist,
                    (lm.z - wrist.z) / dist
                ])

            # Padding second hand
            lm_list.extend([0.0] * 63)

            sequence.append(lm_list)

            mp_draw.draw_landmarks(
                display_frame,
                res_hands,
                mp_hands.HAND_CONNECTIONS
            )

        # ==============================
        # Prediction Logic
        # ==============================

        if frame_counter % 2 == 0 and len(sequence) == SEQ_LENGTH:

            seq_input = np.expand_dims(list(sequence), axis=0)

            res = model.predict(seq_input, verbose=0)[0]

            idx = np.argmax(res)
            confidence = res[idx]

            pred_history.append(idx)

            most_common = Counter(pred_history).most_common(1)

            if most_common and confidence > CONF_THRESHOLD:

                common_idx, count = most_common[0]

                if count >= STABILITY_FRAMES:

                    predicted_word = gesture_labels.get(common_idx, "")

                    current_time = time.time()

                    if (
                        predicted_word != "" and
                        predicted_word != last_word and
                        current_time - last_prediction_time > COOLDOWN_TIME
                    ):

                        sentence_buffer.append(predicted_word)

                        last_word = predicted_word
                        last_prediction_time = current_time

                        sequence.clear()
                        pred_history.clear()

        # ==============================
        # UI Display
        # ==============================

        text_display = " ".join(sentence_buffer[-5:])

        cv2.rectangle(display_frame, (0, h-60), (w, h), (0, 0, 0), -1)

        cv2.putText(
            display_frame,
            f"Detected: {text_display}",
            (20, h-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Language Translator", display_frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord('c'):
            sentence_buffer = []
            last_word = ""


vs.stop()
cv2.destroyAllWindows()