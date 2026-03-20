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
    """
    Return 63 normalized features for one hand:
    21 landmarks * (x, y, z)
    normalized relative to wrist and scaled by wrist-middle_tip distance
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


def main():
    data = []

    if not os.path.exists(GESTURES_FOLDER):
        print(f"❌ Folder not found: {GESTURES_FOLDER}")
        return

    gesture_folders = sorted([
        g for g in os.listdir(GESTURES_FOLDER)
        if os.path.isdir(os.path.join(GESTURES_FOLDER, g))
    ])

    if len(gesture_folders) == 0:
        print("❌ No gesture folders found inside gestures/")
        return

    print("✅ Found gesture folders:")
    for i, g in enumerate(gesture_folders):
        print(f"   Label {i} -> {g}")

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        for label, gesture_name in enumerate(gesture_folders):
            gesture_path = os.path.join(GESTURES_FOLDER, gesture_name)
            print(f"\n📁 Processing: {gesture_name} -> Label {label}")

            video_files = [
                f for f in os.listdir(gesture_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]

            if len(video_files) == 0:
                print(f"⚠ No videos found in {gesture_name}")
                continue

            for video_file in video_files:
                video_path = os.path.join(gesture_path, video_file)
                print(f"   🎥 {video_file}")

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"   ❌ Could not open video: {video_file}")
                    continue

                frame_count = 0
                saved_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # keep same style as your live pipeline
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    hand1 = [0.0] * 63
                    hand2 = [0.0] * 63
                    detected_hands = []

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            feats = extract_hand_features(hand_landmarks)
                            if feats is not None:
                                wrist_x = hand_landmarks.landmark[0].x
                                detected_hands.append((wrist_x, feats))

                    # stable order: left-to-right on screen
                    detected_hands.sort(key=lambda x: x[0])

                    if len(detected_hands) >= 1:
                        hand1 = detected_hands[0][1]

                    if len(detected_hands) >= 2:
                        hand2 = detected_hands[1][1]

                    row = hand1 + hand2  # 126 features total

                    # save only if at least one hand exists
                    if any(v != 0.0 for v in row):
                        row.append(label)
                        data.append(row)
                        saved_count += 1

                cap.release()
                print(f"   ✅ Frames read: {frame_count}, samples saved: {saved_count}")

    if len(data) == 0:
        print("❌ No samples extracted.")
        return

    # create column names
    columns = []
    for hand_idx in [1, 2]:
        for lm_idx in range(21):
            columns.extend([
                f"hand{hand_idx}_x{lm_idx}",
                f"hand{hand_idx}_y{lm_idx}",
                f"hand{hand_idx}_z{lm_idx}",
            ])
    columns.append("label")

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ CSV Saved:", OUTPUT_CSV)
    print("✅ Total Samples:", len(data))
    print("✅ Unique labels:", np.unique(df["label"].values))


if __name__ == "__main__":
    main()
