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
    df = pd.read_csv(CSV_FILE)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    sequences = []
    labels = []

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        label_data = X[idxs]

        for i in range(len(label_data) - SEQ_LENGTH + 1):
            sequences.append(label_data[i:i + SEQ_LENGTH])
            labels.append(label)

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.int32)

    np.save(X_OUT, X_seq)
    np.save(Y_OUT, y_seq)

    print("✅ Safe Sequences Created")
    print("X shape:", X_seq.shape)
    print("y shape:", y_seq.shape)
    print("Unique labels:", np.unique(y_seq))

if __name__ == "__main__":
    main()