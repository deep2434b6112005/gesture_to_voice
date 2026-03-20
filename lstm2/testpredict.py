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
MODEL_PATH = "gesture_gru_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
    print("✅ Model input shape:", model.input_shape)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()


# ==============================
# 3. MediaPipe Setup
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==============================
# 4. Settings & Hyperparameters
# ==============================
# Take sequence length from model if possible
try:
    SEQ_LENGTH = int(model.input_shape[1])
    FEATURES = int(model.input_shape[2])
except Exception:
    SEQ_LENGTH = 30
    FEATURES = 126

sequence = deque(maxlen=SEQ_LENGTH)
sentence_buffer = []

pred_history = deque(maxlen=15)

CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.15
STABILITY_FRAMES = 10

last_word = ""
last_prediction_time = 0
COOLDOWN_TIME = 1.2

frame_counter = 0

# Change these labels to exactly match your training folders
gesture_labels = {
    0: "I am",
    1: "Hungry",
    2: "Want",
    3: "Some water",
    4: "Look",
    5: "Beautiful",
    6: "Help me",
    7: "How many",
    8: "To buy"
    # 9: "Sorry"   # add only if you trained 10 gestures
}

CAMERA_SOURCE = "http://10.31.181.196:8080/video"


# ==============================
# 5. Feature Extraction
# ==============================
def extract_one_hand_features(hand_landmarks):
    """
    63 features for one hand
    normalized to wrist and scaled by wrist-middle_tip distance
    """
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]

    scale = np.sqrt(
        (middle_tip.x - wrist.x) ** 2 +
        (middle_tip.y - wrist.y) ** 2
    )

    if scale == 0:
        return None

    lm_list = []
    for lm in hand_landmarks.landmark:
        lm_list.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    return lm_list  # 63


def extract_features_two_hand(results):
    """
    Returns:
        features: np.array of shape (126,)
        detected_hands: list of (wrist_x, feats, hand_landmarks)
    """
    hand1 = [0.0] * 63
    hand2 = [0.0] * 63

    detected_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            feats = extract_one_hand_features(hand_landmarks)
            if feats is not None:
                wrist_x = hand_landmarks.landmark[0].x
                detected_hands.append((wrist_x, feats, hand_landmarks))

    # Stable order: left-to-right on screen
    detected_hands.sort(key=lambda x: x[0])

    if len(detected_hands) >= 1:
        hand1 = detected_hands[0][1]
    if len(detected_hands) >= 2:
        hand2 = detected_hands[1][1]

    row = hand1 + hand2  # 126
    return np.array(row, dtype=np.float32), detected_hands


# ==============================
# 6. Start Stream
# ==============================
vs = VideoStream(CAMERA_SOURCE)


# ==============================
# 7. Main Loop
# ==============================
with mp_hands.Hands(
    max_num_hands=2,
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

        detected_hands = []

        if results.multi_hand_landmarks:
            feats, detected_hands = extract_features_two_hand(results)

            # draw all detected hands
            for i, (_, _, hand_landmarks) in enumerate(detected_hands[:2]):
                mp_draw.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)

                cv2.putText(
                    display_frame,
                    f"Hand{i+1}",
                    (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            if len(feats) == FEATURES:
                sequence.append(feats)
        else:
            # clear stale sequences when no hand
            sequence.clear()
            pred_history.clear()

        # ==============================
        # Prediction Logic
        # ==============================
        top_text = ""

        if frame_counter % 2 == 0 and len(sequence) == SEQ_LENGTH:
            seq_input = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

            try:
                res = model.predict(seq_input, verbose=0)[0]

                top2 = np.argsort(res)[-2:][::-1]
                idx1 = int(top2[0])
                conf1 = float(res[idx1])

                if len(top2) > 1:
                    idx2 = int(top2[1])
                    conf2 = float(res[idx2])
                else:
                    idx2 = idx1
                    conf2 = 0.0

                word1 = gesture_labels.get(idx1, f"class_{idx1}")
                word2 = gesture_labels.get(idx2, f"class_{idx2}")

                top_text = f"Top1: {word1} ({conf1:.2f}) | Top2: {word2} ({conf2:.2f})"

                if conf1 >= CONF_THRESHOLD and (conf1 - conf2) >= MARGIN_THRESHOLD:
                    pred_history.append(idx1)
                else:
                    pred_history.clear()

                most_common = Counter(pred_history).most_common(1)

                if most_common:
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

            except Exception as e:
                print("❌ Prediction error:", e)

        # ==============================
        # UI Display
        # ==============================
        text_display = " ".join(sentence_buffer[-5:])

        cv2.rectangle(display_frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            f"Hands: {len(detected_hands)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if top_text:
            cv2.putText(
                display_frame,
                top_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        cv2.rectangle(display_frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            f"Detected: {text_display if text_display else '(waiting...)'}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Language Translator - 2 Hand Test", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord('c'):
            sentence_buffer.clear()
            sequence.clear()
            pred_history.clear()
            last_word = ""
            print("🧹 Cleared")

vs.stop()
cv2.destroyAllWindows()