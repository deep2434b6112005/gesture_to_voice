# ==============================
# prediction.py (FINAL) - TFLITE ONLY + FIREBASE + WRONG GESTURE REDUCTION
# VoxBridge - Finger Mode + Model Mode + Firebase
#
# ✅ F mode kept working
# ✅ M mode uses simpler old-style logic
# ✅ Fixed sequence length from model automatically
# ✅ Reduced wrong gesture detection
# ✅ Firebase writes kept
# ✅ Sends mode + gesture_id
# ✅ Editable custom phrases sync
# ✅ Phrase combiner: "I am" + "Hungry" -> "I am hungry"
# ✅ Added margin check
# ✅ Clears stale sequence when hand disappears
# ✅ Better pending phrase flush
# ==============================

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque, Counter
import tensorflow as tf
import multiprocessing as mpx
from queue import Full

# =====================================================
# ✅ YOUR UID
# =====================================================
UID = "tAe0jxFaSTT6CnEZyoY8woYdc9m1"

# =====================================================
# ✅ CAMERA URL
# Set USE_IP_CAMERA = False to use laptop webcam
# =====================================================
USE_IP_CAMERA = True
URL = "http://10.31.181.196:8080/video" if USE_IP_CAMERA else 0

# =====================================================
# ✅ FIREBASE SERVICE ACCOUNT
# =====================================================
SERVICE_ACCOUNT_PATH = r"C:\Users\mharr\Desktop\lstm\serviceAccountKey.json"

# =====================================================
# ✅ MODEL PATH
# =====================================================
MODEL_PATH = "gesture_gru_model.tflite"

# =====================================================
# ✅ LOAD TFLITE MODEL
# =====================================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_INPUT_SHAPE = input_details[0]["shape"]
SEQ_LENGTH = int(MODEL_INPUT_SHAPE[1])

print("✅ TFLite model loaded successfully.")
print("✅ Model input shape:", MODEL_INPUT_SHAPE)
print("✅ Using SEQ_LENGTH =", SEQ_LENGTH)

# =====================================================
# ✅ LABELS
# =====================================================
gesture_labels = {
    0: "I am",
    1: "Hungry",
    2: "Want",
    3: "Some water",
    4: "Look",
    5: "Beautiful",
    6: "Help me",
    7: "How many",
    8: "To buy",
    9: "Sorry"
}

# =====================================================
# ✅ MODEL MODE SETTINGS
# tuned to reduce wrong detections
# =====================================================
PRED_EVERY_N_FRAMES = 2
CONF_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.18
STABILITY_FRAMES = 8
COOLDOWN_TIME = 1.4

sequence = deque(maxlen=SEQ_LENGTH)
pred_history = deque(maxlen=20)

frame_counter = 0
last_prediction_time = 0.0
last_word = ""

SHOW_FPS = True
MAX_WORDS_ON_SCREEN = 5
DEBUG_TOP2 = True

# =====================================================
# ✅ MODE SWITCH
# =====================================================
MODE_FINGER = "finger"
MODE_MODEL = "model"
current_mode = MODE_FINGER

# =====================================================
# ✅ Finger phrases
# =====================================================
finger_map = {
    1: "Hello, how can I help you?",
    2: "The price is 50 rupees.",
    3: "Please wait a moment.",
    4: "Do you need a bag?",
    5: "Thank you, please come again.",
    0: "Total amount is 100 rupees."
}

FINGER_HOLD_FRAMES = 4
FINGER_COOLDOWN_FRAMES = 10
FINGER_CONF = 1.0

# =====================================================
# ✅ Phrase combiner
# =====================================================
pending_isl_word = None
pending_isl_ts = 0.0
pending_isl_conf = 0.0
pending_isl_id = -1
COMBINE_WINDOW_SEC = 1.8
PENDING_FLUSH_SEC = 1.2

# =====================================================
# MEDIAPIPE
# =====================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =====================================================
# THREAD STATE
# =====================================================
lock = threading.Lock()
latest_frame = None
render_frame = None
stop_flag = False

# =====================================================
# PREDICTION STATE - FINGER
# =====================================================
finger_hold = 0
finger_last = -1
finger_cooldown = 0
finger_now = 0

sentence = []

# =====================================================
# FIREBASE de-dup/throttle
# =====================================================
MIN_PUSH_INTERVAL = 0.35
last_sent_word = ""
last_push_ts = 0.0

# =====================================================
# FIREBASE WORKER
# =====================================================
firebase_queue = mpx.Queue(maxsize=80)

def firebase_worker(q, uid: str, service_account_path: str):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("✅ Firebase connected (worker). UID =", uid)

    except Exception as e:
        print("❌ Firebase init error:", e)
        return

    while True:
        item = q.get()
        if item is None:
            break

        word, conf, mode, gesture_id = item

        try:
            db.collection("voxbridge_live").document(uid).set({
                "uid": uid,
                "text": word,
                "mode": str(mode),
                "gesture_id": int(gesture_id),
                "confidence": float(conf),
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })

            db.collection("gestures").document("latest").set({
                "uid": uid,
                "text": word,
                "mode": str(mode),
                "gesture_id": int(gesture_id),
                "confidence": float(conf),
                "processed": False,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })

        except Exception as e:
            print("❌ Firestore write failed:", e)

def send_firebase(word, conf, mode, gesture_id):
    global last_push_ts, last_sent_word
    now = time.time()

    if word == last_sent_word:
        return
    if (now - last_push_ts) < MIN_PUSH_INTERVAL:
        return

    try:
        firebase_queue.put_nowait((word, float(conf), mode, int(gesture_id)))
        last_push_ts = now
        last_sent_word = word
        print(f"📤 Firestore sent: {word} | mode={mode} | id={gesture_id} | conf={conf:.2f}")
    except Full:
        pass
    except Exception:
        pass

# =====================================================
# ✅ Firebase phrase sync
# =====================================================
def start_phrase_sync_thread():
    def loop():
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
                firebase_admin.initialize_app(cred)

            db = firestore.client()
            print("✅ Firebase connected (sync thread). UID =", UID)
        except Exception as e:
            print("❌ Phrase sync init error:", e)
            return

        last_sync = 0.0
        while not stop_flag:
            try:
                if (time.time() - last_sync) >= 2.5:
                    doc = db.collection("users").document(UID) \
                        .collection("gestures").document("my_gestures").get()

                    if doc.exists:
                        data = doc.to_dict() or {}
                        for i in range(6):
                            k = f"gesture_{i}_phrase"
                            v = data.get(k, None)
                            if isinstance(v, str) and v.strip():
                                finger_map[i] = v.strip()

                    last_sync = time.time()
            except Exception as e:
                print("⚠ Phrase sync error:", e)

            time.sleep(0.2)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

# =====================================================
# CAMERA THREAD
# =====================================================
def camera_loop():
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(URL)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("❌ Cannot open camera source:", URL)
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        with lock:
            latest_frame = frame

    cap.release()

# =====================================================
# ✅ Feature extraction
# =====================================================
def extract_features_old_style(hand):
    wrist = hand.landmark[0]

    dist = np.sqrt(
        (hand.landmark[9].x - wrist.x) ** 2 +
        (hand.landmark[9].y - wrist.y) ** 2
    )

    if dist == 0:
        dist = 1.0

    lm_list = []
    for lm in hand.landmark:
        lm_list.extend([
            (lm.x - wrist.x) / dist,
            (lm.y - wrist.y) / dist,
            (lm.z - wrist.z) / dist
        ])

    # Keep same feature size as training input (126)
    lm_list.extend([0.0] * 63)
    return np.array(lm_list, dtype=np.float32)

# =====================================================
# FINGER COUNT HELPERS
# =====================================================
def is_thumb_up(lm):
    return lm[4].x > lm[3].x

def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    count = 0
    if is_thumb_up(lm):
        count += 1

    for tip, pip in zip(tips[1:], pips[1:]):
        if lm[tip].y < lm[pip].y:
            count += 1

    return int(count)

# =====================================================
# ✅ TFLite prediction helper
# =====================================================
def predict_tflite(seq_input):
    expected_shape = input_details[0]["shape"]

    if seq_input.shape[1] != expected_shape[1]:
        print(f"❌ Sequence mismatch: got {seq_input.shape[1]}, expected {expected_shape[1]}")
        return None

    interpreter.set_tensor(input_details[0]["index"], seq_input.astype(np.float32))
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]["index"])[0]
    return res.astype(np.float32)

# =====================================================
# Phrase combiner
# =====================================================
def handle_isl_output(word, conf, gesture_id):
    global pending_isl_word, pending_isl_ts, pending_isl_conf, pending_isl_id

    now = time.time()

    if pending_isl_word is not None and (now - pending_isl_ts) >= PENDING_FLUSH_SEC:
        send_firebase(pending_isl_word, pending_isl_conf, "isl", pending_isl_id)
        pending_isl_word = None
        pending_isl_ts = 0.0
        pending_isl_conf = 0.0
        pending_isl_id = -1

    if word == "I am":
        pending_isl_word = "I am"
        pending_isl_ts = now
        pending_isl_conf = conf
        pending_isl_id = gesture_id
        return

    if word == "Hungry" and pending_isl_word == "I am" and (now - pending_isl_ts) <= COMBINE_WINDOW_SEC:
        combined = "I am hungry"
        send_firebase(combined, min(1.0, max(pending_isl_conf, conf)), "isl", 1001)
        pending_isl_word = None
        pending_isl_ts = 0.0
        pending_isl_conf = 0.0
        pending_isl_id = -1
        return

    send_firebase(word, conf, "isl", gesture_id)

# =====================================================
# AI THREAD
# =====================================================
def ai_loop():
    global latest_frame, render_frame, stop_flag
    global current_mode
    global finger_hold, finger_last, finger_cooldown, finger_now
    global frame_counter, last_prediction_time, last_word
    global pending_isl_word, pending_isl_ts, pending_isl_conf, pending_isl_id

    fps_timer = time.time()
    fps_counter = 0
    fps_val = 0.0
    last_debug_print = 0.0

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while not stop_flag:
            try:
                with lock:
                    frame = None if latest_frame is None else latest_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                display = frame.copy()
                frame_counter += 1
                fps_counter += 1

                now = time.time()
                if now - fps_timer >= 1.0:
                    fps_val = fps_counter / (now - fps_timer)
                    fps_timer = now
                    fps_counter = 0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                hand = None
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

                top_label = ""
                top_conf = 0.0

                # =============================================
                # F MODE
                # =============================================
                if current_mode == MODE_FINGER:
                    finger_now = 0
                    if hand is not None:
                        finger_now = count_fingers(hand)
                    else:
                        finger_hold = max(0, finger_hold - 1)

                    if finger_cooldown > 0:
                        finger_cooldown -= 1
                        finger_hold = 0
                        finger_last = -1
                    else:
                        if finger_now == finger_last:
                            finger_hold += 1
                        else:
                            finger_last = finger_now
                            finger_hold = 1

                        if finger_hold >= FINGER_HOLD_FRAMES:
                            phrase = finger_map.get(finger_now, "")
                            if phrase and phrase != last_word:
                                sentence.append(phrase)
                                last_word = phrase
                                send_firebase(phrase, FINGER_CONF, "custom", finger_now)

                            finger_hold = 0
                            finger_last = -1
                            finger_cooldown = FINGER_COOLDOWN_FRAMES

                    top_label = f"Fingers: {finger_now}"
                    top_conf = 1.0

                # =============================================
                # M MODE - stricter logic to reduce wrong gesture
                # =============================================
                else:
                    if hand is not None:
                        feats = extract_features_old_style(hand)
                        sequence.append(feats)
                    else:
                        pred_history.clear()
                        sequence.clear()

                    if frame_counter % PRED_EVERY_N_FRAMES == 0 and len(sequence) == SEQ_LENGTH:
                        seq_input = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                        res = predict_tflite(seq_input)

                        if res is not None:
                            top2 = np.argsort(res)[-2:][::-1]
                            idx1 = int(top2[0])
                            idx2 = int(top2[1])
                            conf1 = float(res[idx1])
                            conf2 = float(res[idx2])

                            top_label = f"{gesture_labels.get(idx1, '?')} / {gesture_labels.get(idx2, '?')}"
                            top_conf = conf1

                            if DEBUG_TOP2 and (time.time() - last_debug_print) > 0.5:
                                print(
                                    f"DEBUG -> top1: {gesture_labels.get(idx1, '?')} ({conf1:.2f}) | "
                                    f"top2: {gesture_labels.get(idx2, '?')} ({conf2:.2f})"
                                )
                                last_debug_print = time.time()

                            # only strong and clearly separated predictions go into history
                            if conf1 >= CONF_THRESHOLD and (conf1 - conf2) >= MARGIN_THRESHOLD:
                                pred_history.append(idx1)
                            else:
                                pred_history.clear()

                            if len(pred_history) > 0:
                                most_common = Counter(pred_history).most_common(1)
                            else:
                                most_common = []

                            if most_common:
                                common_idx, count = most_common[0]

                                if count >= STABILITY_FRAMES:
                                    predicted_word = gesture_labels.get(common_idx, "")
                                    current_time = time.time()

                                    if (
                                        predicted_word != "" and
                                        predicted_word != last_word and
                                        (current_time - last_prediction_time) > COOLDOWN_TIME
                                    ):
                                        sentence.append(predicted_word)
                                        last_word = predicted_word
                                        last_prediction_time = current_time

                                        handle_isl_output(predicted_word, conf1, common_idx)

                                        sequence.clear()
                                        pred_history.clear()

                    # Flush pending phrase if needed
                    if pending_isl_word is not None:
                        if (time.time() - pending_isl_ts) >= PENDING_FLUSH_SEC:
                            send_firebase(pending_isl_word, pending_isl_conf, "isl", pending_isl_id)
                            pending_isl_word = None
                            pending_isl_ts = 0.0
                            pending_isl_conf = 0.0
                            pending_isl_id = -1

                # =============================================
                # UI
                # =============================================
                text = " ".join(sentence[-MAX_WORDS_ON_SCREEN:])
                h, w = display.shape[:2]

                cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
                mode_txt = "FINGER" if current_mode == MODE_FINGER else "MODEL (TFLITE)"
                cv2.putText(display, f"VOXBRIDGE • LIVE • MODE: {mode_txt}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                if top_label:
                    cv2.putText(display, f"Top: {top_label}  conf={top_conf:.2f}",
                                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                cv2.rectangle(display, (0, h - 70), (w, h), (0, 0, 0), -1)
                cv2.putText(display, f"Detected: {text if text else '(waiting...)'}",
                            (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2)

                if SHOW_FPS:
                    cv2.putText(display, f"FPS: {fps_val:.1f}",
                                (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                cv2.putText(display, "ESC Quit | C Clear | F Finger | M Model", (15, h - 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                with lock:
                    render_frame = display

            except Exception as e:
                print("❌ ai_loop error:", e)
                time.sleep(0.1)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    mpx.freeze_support()

    start_phrase_sync_thread()

    fb_proc = mpx.Process(
        target=firebase_worker,
        args=(firebase_queue, UID, SERVICE_ACCOUNT_PATH),
        daemon=True
    )
    fb_proc.start()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    ai_thread = threading.Thread(target=ai_loop, daemon=True)

    cam_thread.start()
    ai_thread.start()

    print("🎥 Camera:", URL)
    print("✅ Writing to Firestore:")
    print("   1) voxbridge_live /", UID)
    print("   2) gestures / latest")
    print("✅ Controls: ESC quit, C clear, F finger mode, M model mode")

    while True:
        with lock:
            show = None if render_frame is None else render_frame.copy()

        if show is not None:
            cv2.imshow("VoxBridge (Finger + Model + Firebase)", show)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            stop_flag = True
            break

        if key == ord('c'):
            with lock:
                sentence.clear()
                sequence.clear()
                pred_history.clear()
                pending_isl_word = None
                pending_isl_ts = 0.0
                pending_isl_conf = 0.0
                pending_isl_id = -1
                last_word = ""
            print("🧹 Cleared sentence and model buffers")

        if key == ord('f'):
            current_mode = MODE_FINGER
            print("✅ Mode switched to: FINGER")

            with lock:
                sequence.clear()
                pred_history.clear()
                pending_isl_word = None
                pending_isl_ts = 0.0
                pending_isl_conf = 0.0
                pending_isl_id = -1
                last_word = ""

            finger_hold = 0
            finger_last = -1
            finger_cooldown = 0

        if key == ord('m'):
            current_mode = MODE_MODEL
            print("✅ Mode switched to: MODEL (TFLite)")

            with lock:
                sequence.clear()
                pred_history.clear()
                pending_isl_word = None
                pending_isl_ts = 0.0
                pending_isl_conf = 0.0
                pending_isl_id = -1
                last_word = ""

            finger_hold = 0
            finger_last = -1
            finger_cooldown = 0

    try:
        firebase_queue.put(None)
    except Exception:
        pass

    cv2.destroyAllWindows()
