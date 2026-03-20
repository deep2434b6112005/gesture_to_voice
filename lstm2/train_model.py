import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
X_FILE = "X.npy"
Y_FILE = "y.npy"
MODEL_OUT = "gesture_gru_model.h5"

EPOCHS = 60
BATCH_SIZE = 32
SEQ_LENGTH = 30
FEATURES = 126


def main():
    try:
        X = np.load(X_FILE)
        y = np.load(Y_FILE)
    except Exception as e:
        print(f"❌ Failed to load training files: {e}")
        return

    print("✅ Loaded training data")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Unique labels:", np.unique(y))

    if len(X.shape) != 3:
        print("❌ X must be 3D: (samples, seq_len, features)")
        return

    if X.shape[1] != SEQ_LENGTH or X.shape[2] != FEATURES:
        print(f"❌ Expected X shape (_, {SEQ_LENGTH}, {FEATURES}) but got {X.shape}")
        return

    num_classes = len(np.unique(y))
    print("✅ Number of classes detected:", num_classes)

    y_cat = to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Sequential([
        Input(shape=(SEQ_LENGTH, FEATURES)),

        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        GRU(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-4,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_OUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop, checkpoint]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    model.save(MODEL_OUT)
    print("✅ Model saved as", MODEL_OUT)


if __name__ == "__main__":
    main()