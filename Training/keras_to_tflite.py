import tensorflow as tf

model = tf.keras.models.load_model('traffic_sign_identification.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('converted_model.tflite', 'wb') as f:
  f.write(tflite_model)