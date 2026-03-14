import tensorflow as tf

# Load model (works in this env)
model = tf.keras.models.load_model("gesture_gru_model.h5", compile=False)

# Save as TensorFlow SavedModel
model.export("saved_gesture_model")

print("✅ SavedModel exported")

# Convert SavedModel → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_gesture_model")

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

with open("gesture_gru_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved")