import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

save_dir = "/Users/cooperwang/PycharmProjects/TrafficSigns"
model_save_path = os.path.join(save_dir, "traffic_sign_model")

def load_data(data_dir):

    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels


train_data_dir = "/Users/cooperwang/PycharmProjects/TrafficSigns/traffic/datasets/Training"
test_data_dir = "/Users/cooperwang/PycharmProjects/TrafficSigns/traffic/datasets/Testing"

images, labels = load_data(train_data_dir)

test_images, test_labels = load_data(test_data_dir)

images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]

test_images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in test_images]

labels_a = np.array(labels)
images_a = np.array(images32)

test_labels = np.array(test_labels)
test_images32 = np.array(test_images32)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(62)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(images_a, labels_a, epochs=50)

# model.save('traffic_sign_identification.keras')

model.evaluate(test_images32,  test_labels, verbose=2)
