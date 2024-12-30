import os
import cv2
import onnxruntime as ort
import numpy as np
import time
import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

import os
import cv2
import onnxruntime as ort
import numpy as np

# def preprocess_image(image_path, input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
def preprocess_image(image_path, input_size, mean=(0.485, 0.456, 0.406), std=(1.0, 1.0, 1.0)):
    """
    Preprocesses the input image for the ONNX model.
    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Expected input size of the model (H, W).
    Returns:
        np.ndarray: Preprocessed image.
    """
    # original code without normalization

    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # if image is None:
    #     raise FileNotFoundError(f"Image file '{image_path}' not found or cannot be read.")
    # image = cv2.resize(image, input_size)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # image = np.transpose(image, (2, 0, 1))
    # return np.expand_dims(image, axis=0)

    # normalize for all 3 channels
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or cannot be read.")
    
    # Resize and convert color format
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Normalize with mean and std
    image = (image - mean) / std

    # Convert to the required shape
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

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
            # start timer
            start_time = time.time()
            # # Preprocess the image
            input_image = preprocess_image(image_path, input_size)
            
            # # Run the ONNX model
            inputs = {session.get_inputs()[0].name: input_image}
            outputs = session.run(None, inputs)

            output_mask = np.squeeze(outputs[0])  # Remove batch dimension
            output_mask = (output_mask * 255).astype(np.uint8)  # Scale mask to [0, 255]

            # Threshold the mask for binary segmentation
            _, binary_mask = cv2.threshold(output_mask, 128, 255, cv2.THRESH_BINARY)

            # Load the original image for applying the mask
            original_image = cv2.imread(image_path)
            original_image = cv2.resize(original_image, input_size)

            # Create the final output: background black, foreground visible
            output_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

            end_time = time.time()
            elapsed_time = end_time - start_time

            filename_wo_ext, ext = os.path.splitext(filename)
            output_filename = f"{filename_wo_ext}_{elapsed_time:.2f}s{ext}"  # Append time in seconds
            output_path = os.path.join(output_images_dir, output_filename)

            cv2.imwrite(output_path, output_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Parameters
onnx_model_path = "./models/isnet-general.onnx"  # Path to the ONNX model
input_images_dir = "./input_images"  # Directory containing input images
output_base_dir = "./output_images"  # Base directory for all output images

run_onnx_model(onnx_model_path, input_images_dir, output_base_dir)
