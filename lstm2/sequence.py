import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
CSV_FILE = "hand_landmarks_labeled.csv"
SEQ_LENGTH = 30
X_OUT = "X.npy"
Y_OUT = "y.npy"


def main():
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"❌ Failed to read {CSV_FILE}: {e}")
        return

    if df.shape[1] < 2:
        print("❌ CSV format looks invalid.")
        return

    X = df.iloc[:, :-1].values.astype(np.float32)   # 126 features
    y = df.iloc[:, -1].values.astype(np.int32)

    print("✅ Loaded CSV")
    print("X raw shape:", X.shape)
    print("y raw shape:", y.shape)
    print("Unique labels:", np.unique(y))

    sequences = []
    labels = []

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        label_data = X[idxs]

        if len(label_data) < SEQ_LENGTH:
            print(f"⚠ Skipping label {label} because samples < SEQ_LENGTH")
            continue

        for i in range(len(label_data) - SEQ_LENGTH + 1):
            seq = label_data[i:i + SEQ_LENGTH]
            sequences.append(seq)
            labels.append(label)

    if len(sequences) == 0:
        print("❌ No valid sequences created.")
        return

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.int32)

    np.save(X_OUT, X_seq)
    np.save(Y_OUT, y_seq)

    print("\n✅ Safe Sequences Created")
    print("X shape:", X_seq.shape)   # expected: (samples, 30, 126)
    print("y shape:", y_seq.shape)
    print("Unique labels:", np.unique(y_seq))


if __name__ == "__main__":
    main()