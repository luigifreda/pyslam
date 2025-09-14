import sys
import pyslam.config as config

config.cfg.set_lib("lfnet", prepend=True)

import tensorflow as tf
import numpy as np

tf.get_logger().setLevel("ERROR")

lfnet_base_path = "../../thirdparty/lfnet"
lfnet_model_path = lfnet_base_path + "/pretrained/lfnet-norotaug"
lfnet_checkpoint_path = lfnet_base_path + "/pretrained/lfnet-norotaug"

MODEL_PATH = lfnet_base_path + "/models"
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def load_pretrained_network(checkpoint_path):
    try:
        # Create a new model instance
        model = tf.keras.models.Sequential()  # Replace with your model architecture

        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(model=model)

        # Restore the checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
        return model
    except tf.errors.NotFoundError as e:
        if "bn1/beta" in str(e):
            print(
                "Warning: Key 'bn1/beta' not found in checkpoint. Please check the checkpoint file."
            )
        else:
            print("Error loading model:", e)
        return None


if __name__ == "__main__":
    model = load_pretrained_network(lfnet_checkpoint_path)
    if model:
        print("Model loaded successfully.")

        # Create some dummy data to test the model
        dummy_input = np.random.rand(1, 224, 224, 3).astype(
            np.float32
        )  # Adjust the shape as per your model's input requirements

        # Perform inference
        predictions = model(dummy_input)
        print("Predictions:", predictions)
