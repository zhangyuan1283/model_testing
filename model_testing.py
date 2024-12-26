import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm


def load_tflite_model(model_path):
    """
    Load the TFLite model into an interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# def preprocess_image(image, input_shape):
#     """
#     Preprocess the image to match the model's expected input:
#     - Resize the image to (input_shape[1], input_shape[2]).
#     - Ensure 3 channels (RGB).
#     - Normalize pixel values to [0, 1].
#     """
#     # Resize the image (width, height as per input_shape)
#     resized_image = cv2.resize(image, (input_shape[2], input_shape[1]))  # Model expects (width, height)

#     # Ensure image has 3 channels (convert grayscale to RGB if needed)
#     if len(resized_image.shape) != 3 or resized_image.shape[-1] != 3:
#         resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

#     # Normalize pixel values to [0, 1]
#     normalized_image = resized_image.astype(np.float32) / 255.0

#     # Add batch dimension to match [1, height, width, channels]
#     return np.expand_dims(normalized_image, axis=0)

def preprocess_image(image, input_shape):
    """
    Preprocess the image to match the model's expected input:
    - Resize the image to (input_shape[2], input_shape[3]) for height and width.
    - Ensure 3 channels (RGB).
    - Normalize pixel values to [0, 1].
    - Reshape the tensor into [1, channels, height, width].
    """
    # Resize the image (height, width as per input_shape)
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))  # Model expects (width, height)

    # Ensure the image has 3 channels (convert grayscale to RGB if needed)
    if len(resized_image.shape) != 3 or resized_image.shape[-1] != 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Transpose dimensions to [height, width, channels] -> [channels, height, width]
    # transposed_image = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension to match [1, channels, height, width]
    # return np.expand_dims(transposed_image, axis=0)
    return np.expand_dims(normalized_image, axis=0)



# def postprocess_mask(mask, original_shape):
#     """
#     Postprocess the segmentation mask:
#     - Resize to original image shape.
#     - Threshold to create a binary mask.
#     """
#     mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
#     binary_mask = (mask > 0.5).astype(np.uint8)  # Thresholding
#     return binary_mask

def postprocess_mask(mask, original_shape):
    """
    Postprocess the segmentation mask:
    - Remove batch dimension.
    - Resize to match the original image shape.
    - Threshold to create a binary mask.
    """
    print(f"Original mask shape: {mask.shape}")

    # Remove batch dimension (mask is [1, height, width])
    if len(mask.shape) == 3:
        mask = mask[0]

    print(f"Mask shape after removing batch dimension: {mask.shape}")

    # Resize mask to the original image dimensions (height, width)
    resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]))  # (width, height)
    print(f"Resized mask shape: {resized_mask.shape}")

    # Threshold to create a binary mask
    binary_mask = (resized_mask > 0.5).astype(np.uint8)  # Ensure binary mask
    print(f"Binary mask shape: {binary_mask.shape}")

    return binary_mask



# def remove_background(image, mask):
#     """
#     Apply the mask to the image to remove the background.
#     """
#     transparent_image = image.copy()
#     transparent_image[mask == 0] = 0  # Set background pixels to black
#     return transparent_image

def remove_background(image, mask):
    """
    Apply the binary mask to the image to remove the background.
    - Expand mask dimensions to match the image.
    - Apply the mask to set background pixels to black.
    """
    # Ensure the mask has a single channel
    if len(mask.shape) == 2:  # 2D binary mask
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension (height, width, 1)

    # Tile the mask to match image channels (height, width, 3)
    mask = np.tile(mask, (1, 1, 3))

    print(f"Image shape: {image.shape}")
    print(f"Mask shape after tiling: {mask.shape}")

    # Apply the mask to the image
    transparent_image = image.copy()
    transparent_image[mask == 0] = 0  # Set background pixels to black
    return transparent_image


def batch_process_images(input_folder, output_folder, model_path):
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    print(f"Model expects input shape: {input_shape}")

    os.makedirs(output_folder, exist_ok=True)

    for image_name in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_name}. Skipping.")
            continue

        input_image = preprocess_image(image, input_shape)

        # Debug: Ensure input tensor matches expected dimensions
        print(f"Processing {image_name} with input tensor shape: {input_image.shape}")

        try:
            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            mask = postprocess_mask(output_data[0], image.shape[:3])
            result_image = remove_background(image, mask)
            cv2.imwrite(output_path, result_image)
        except ValueError as e:
            print(f"Error processing {image_name}: {e}")



if __name__ == "__main__":
    # Configuration
    model_path = "models/isnet-general-use_float32.tflite"  # Replace with your TFLite model path
    input_folder = "input_images"  # Replace with your input images folder
    output_folder = "output_images"  # Replace with your output images folder

    # Run batch processing
    batch_process_images(input_folder, output_folder, model_path)
