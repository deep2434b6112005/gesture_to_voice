import tensorflow as tf
import os

H5_MODEL = "gesture_gru_model.h5"
SAVED_MODEL_DIR = "saved_gesture_model"
TFLITE_OUT = "gesture_gru_model.tflite"


def main():
    if not os.path.exists(H5_MODEL):
        print(f"❌ Model file not found: {H5_MODEL}")
        return

    print("✅ Loading model:", H5_MODEL)
    model = tf.keras.models.load_model(H5_MODEL, compile=False)

    print("✅ Exporting SavedModel...")
    model.export(SAVED_MODEL_DIR)
    print("✅ SavedModel exported to:", SAVED_MODEL_DIR)

    print("✅ Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True

    tflite_model = converter.convert()

    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)

    print("✅ TFLite model saved as:", TFLITE_OUT)


if __name__ == "__main__":
    main()