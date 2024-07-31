import skimage
import tensorflow as tf
import numpy as np

image = (skimage.io.imread("Belgian_traffic_sign_B1.png"))

image = skimage.transform.resize(image, (32, 32, 3), mode='constant')

image = np.float32(image)

image = np.array(image)

image = np.expand_dims(image, axis=0)

new_model = tf.keras.models.load_model("traffic_sign_identification.keras")

print(np.argmax(new_model.predict(image)))