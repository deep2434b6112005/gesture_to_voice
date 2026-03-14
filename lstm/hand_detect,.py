import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# =========================
# CONFIG
# =========================
GESTURES_FOLDER = "gestures"
OUTPUT_CSV = "hand_landmarks_labeled.csv"

mp_hands = mp.solutions.hands

def extract_hand_features(hand_landmarks):
    """Return 63 features normalized to wrist distance scaling."""
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]

    scale = np.sqrt((middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2)
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

def main():
    data = []

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        for label, gesture_name in enumerate(sorted(os.listdir(GESTURES_FOLDER))):
            gesture_path = os.path.join(GESTURES_FOLDER, gesture_name)
            if not os.path.isdir(gesture_path):
                continue

            print(f"Processing: {gesture_name} → Label {label}")

            for video_file in os.listdir(gesture_path):
                if not video_file.endswith((".mp4", ".avi", ".mov")):
                    continue

                cap = cv2.VideoCapture(os.path.join(gesture_path, video_file))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    hands_data = {"Right": [0.0]*63, "Left": [0.0]*63}

                    if results.multi_hand_landmarks and results.multi_handedness:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness
                        ):
                            hand_type = handedness.classification[0].label  # "Right"/"Left"
                            feats = extract_hand_features(hand_landmarks)
                            if feats is not None:
                                hands_data[hand_type] = feats

                    row = hands_data["Right"] + hands_data["Left"]  # 126
                    if any(v != 0.0 for v in row):
                        row.append(label)
                        data.append(row)

                cap.release()

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ CSV Saved:", OUTPUT_CSV)
    print("Total Samples:", len(data))
    if len(data) > 0:
        print("Unique labels:", np.unique(df.iloc[:, -1]))

if __name__ == "__main__":
    main()
