import os
import cv2
import onnxruntime as ort
import numpy as np

import os
import cv2
import onnxruntime as ort
import numpy as np

def preprocess_image(image_path, input_size):
    """
    Preprocesses the input image for the ONNX model.
    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Expected input size of the model (H, W).
    Returns:
        np.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or cannot be read.")
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)

def get_model_input_size(onnx_model_path):
    """
    Extracts the input size from the ONNX model.
    Args:
        onnx_model_path (str): Path to the ONNX model.
    Returns:
        tuple: Input size (height, width).
    """
    session = ort.InferenceSession(onnx_model_path)
    input_shape = session.get_inputs()[0].shape
    # Assuming input shape is (N, C, H, W)
    return input_shape[2], input_shape[3]

def run_onnx_model(onnx_model_path, input_images_dir, output_base_dir):
    """
    Runs an ONNX model on a batch of images from a directory and saves the output.
    Args:
        onnx_model_path (str): Path to the ONNX model.
        input_images_dir (str): Directory containing input images.
        output_base_dir (str): Base directory to save output images.
    """
    # Get model name and create model-specific output directory
    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    output_images_dir = os.path.join(output_base_dir, model_name)
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    # Load ONNX model and get input size
    input_size = get_model_input_size(onnx_model_path)
    session = ort.InferenceSession(onnx_model_path)

    # Iterate over all images in the input directory
    for filename in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, filename)
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
            print(f"Skipping non-image file: {filename}")
            continue
        try:
            # Preprocess the image
            input_image = preprocess_image(image_path, input_size)
            
            # Run the ONNX model
            inputs = {session.get_inputs()[0].name: input_image}
            outputs = session.run(None, inputs)
            
            # Postprocess and save the output
            output_image = np.squeeze(outputs[0])  # Remove batch dimension
            output_image = (output_image * 255).astype(np.uint8)  # Scale back to [0, 255]
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            output_path = os.path.join(output_images_dir, filename)
            cv2.imwrite(output_path, output_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Parameters
onnx_model_path = "./models/HRSOD.onnx"  # Path to the ONNX model
input_images_dir = "./input_images"  # Directory containing input images
output_base_dir = "./output_images"  # Base directory for all output images

run_onnx_model(onnx_model_path, input_images_dir, output_base_dir)
