import os
import onnxruntime as ort
import numpy as np
import time
from PIL import Image, ImageOps
from typing import Tuple

def preprocess_image_pillow(pil_image: Image.Image, input_size: Tuple[int, int],
                           mean: Tuple[float, float, float],
                           std: Tuple[float, float, float]) -> np.ndarray:
    """
    Preprocesses the input Pillow image for the ONNX model with normalization.

    Args:
        pil_image (PIL.Image.Image): Input image as a Pillow Image.
        input_size (tuple): Expected input size of the model (W, H).
        mean (tuple): Mean values for each channel (R, G, B).
        std (tuple): Standard deviation values for each channel (R, G, B).

    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    # Resize the image to the model's expected input size
    pil_image = pil_image.resize(input_size, Image.BILINEAR)

    # Convert image to RGB if it's not
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert PIL Image to NumPy array
    image = np.array(pil_image).astype(np.float32)

    # Normalize the image to [0, 1]
    image /= 255.0

    # Apply mean and std normalization
    image -= np.array(mean)
    image /= np.array(std)

    # Transpose to (C, H, W)
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def get_model_input_size(onnx_model_path: str) -> Tuple[int, int]:
    """
    Extracts the input size from the ONNX model.

    Args:
        onnx_model_path (str): Path to the ONNX model.

    Returns:
        tuple: Input size (width, height).
    """
    session = ort.InferenceSession(onnx_model_path)
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic dimensions (e.g., None)
    width = input_shape[3] if input_shape[3] is not None else 512  # Default width
    height = input_shape[2] if input_shape[2] is not None else 512  # Default height
    return width, height

def get_concat_v(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Concatenate two images vertically.

    Args:
        img1 (PIL.Image.Image): The first image.
        img2 (PIL.Image.Image): The second image to be concatenated below the first image.

    Returns:
        PIL.Image.Image: The concatenated image.
    """
    # Ensure both images have the same width
    if img1.width != img2.width:
        # Resize img2 to match img1's width while maintaining aspect ratio
        img2 = img2.resize((img1.width, int(img2.height * img1.width / img2.width)), Image.ANTIALIAS)

    # Create a new image with combined height
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst

def run_onnx_model_pillow(
    onnx_model_path: str,
    input_images_dir: str,
    output_base_dir: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    visualize: bool = False
):
    """
    Runs an ONNX model on a batch of images from a directory and saves the output.

    Args:
        onnx_model_path (str): Path to the ONNX model.
        input_images_dir (str): Directory containing input images.
        output_base_dir (str): Base directory to save output images.
        mean (tuple): Mean values for each channel (R, G, B).
        std (tuple): Standard deviation values for each channel (R, G, B).
        visualize (bool): If True, saves concatenated input and output images for visualization.
    """
    # Get model name and create model-specific output directory
    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    output_images_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(output_images_dir, exist_ok=True)

    # Load ONNX model and get input size
    input_size = get_model_input_size(onnx_model_path)
    session = ort.InferenceSession(onnx_model_path)

    # Iterate over all images in the input directory
    for filename in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue
        try:
            # Start timer
            start_time = time.time()

            # Open the image using Pillow
            with Image.open(image_path) as pil_image:
                # Preprocess the image with normalization
                input_image = preprocess_image_pillow(pil_image, input_size, mean, std)

                # Run the ONNX model
                inputs = {session.get_inputs()[0].name: input_image}
                ort_outs = session.run(None, inputs)

                # Process the output mask
                pred = ort_outs[0][:, 0, :, :]  # Assuming the mask is in the first channel

                # Normalize the mask to [0, 1]
                ma = np.max(pred)
                mi = np.min(pred)
                pred = (pred - mi) / (ma - mi) if ma != mi else pred

                # Squeeze to remove batch and channel dimensions
                pred = np.squeeze(pred)

                # Convert the mask to a PIL Image
                mask_pil = Image.fromarray((pred * 255).astype("uint8"), mode="L").resize(pil_image.size, Image.Resampling.LANCZOS)

                # Threshold the mask for binary segmentation
                binary_mask = mask_pil.point(lambda p: 255 if p > 64 else 0)

                # Create the final output: background black, foreground visible
                original_image = pil_image.convert("RGBA")
                mask_rgba = binary_mask.convert("L")
                output_image = Image.composite(original_image, Image.new("RGBA", pil_image.size, (0, 0, 0, 255)), mask_rgba)

                # Calculate elapsed time
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Prepare output filename
                filename_wo_ext, ext = os.path.splitext(filename)
                output_filename = f"{filename_wo_ext}_s.png"  # Saving as PNG to support transparency
                output_path = os.path.join(output_images_dir, output_filename)

                # Save the processed image
                output_image.save(output_path)
                print(f"Processed and saved: {output_path} in {elapsed_time:.2f} seconds")

                # # Optional: Visualize by concatenating input and output
                # if visualize:
                #     # Resize images for visualization
                #     input_vis = pil_image.convert("RGBA").resize(input_size, Image.Resampling.LANCZOS)
                #     output_vis = output_image.resize(input_size, Image.Resampling.LANCZOS)
                #     concatenated = get_concat_v(input_vis, output_vis)
                #     concat_filename = f"{filename_wo_ext}_concat.png"
                #     concat_path = os.path.join(output_images_dir, concat_filename)
                #     concatenated.save(concat_path)
                #     print(f"Saved concatenated image: {concat_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Parameters
onnx_model_path = "./models/isnet_gen.onnx"  # Path to the ONNX model
input_images_dir = "./input_images"          # Directory containing input images
output_base_dir = "./output_images"          # Base directory for all output images

# Normalization parameters (example values; adjust based on your model's requirements)
mean = (0.485, 0.456, 0.406)  # Mean for RGB channels
std = (1.0, 1.0, 1.0)         # Std for RGB channels (adjusted to (1,1,1) as per original script)

# Run the ONNX model with normalization
run_onnx_model_pillow(
    onnx_model_path=onnx_model_path,
    input_images_dir=input_images_dir,
    output_base_dir=output_base_dir,
    mean=mean,
    std=std,
    visualize=True  # Set to True to save concatenated input and output images
)
