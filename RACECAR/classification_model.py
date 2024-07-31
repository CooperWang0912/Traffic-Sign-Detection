import os
import numpy as np
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'traffic_sign_identification_edgetpu.tflite')

interpreter = None


def load_model(model_path):
    """
    Load the TFLite model and create an interpreter.

    Args:
    model_path (str): Path to the TFLite model file.

    Returns:
    None
    """
    global interpreter

    # Load the TFLite model and allocate tensors
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    print("Model loaded successfully.")


def infer(input_data):
    global interpreter

    if interpreter is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    # Ensure input_data is a 1D numpy array
    input_data = np.array(input_data).flatten()

    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check input shape
    input_shape = input_details[0]['shape']

    # Quantize the input data to INT8

    input_data = input_data.reshape(input_shape)
    # print(f'Input shape: {input_shape}')
    # print(f'Data shape: {input_data.shape}')

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize the output data

    # Process the output data
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]

    return predicted_class, confidence


# Example usage
if __name__ == "__main__":
    # Load the model
    load_model("traffic_sign_identification_edgetpu.tflite")

    # Example input data
    sample = np.random.rand(3072)

    sample = np.float32(sample)

    sample = sample.reshape((32, 32, 3))

    sample = np.expand_dims(sample, axis=0)

    # Perform inference
    predicted_class, confidence = infer(sample)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")