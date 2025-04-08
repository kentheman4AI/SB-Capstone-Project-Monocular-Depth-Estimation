#Please write a script that will:
#1.  prompt the user to select folders for pictures, paintings, depth maps (.png files), addons (images), and an output directory
#2.  For each picture
#a.  randomly select a painting and cartoonize it
#b.  randomly crop and keep between 30-60% of the height, and 50-100% of the width
#c.  then paste this entire painting onto the picture at a random location
#d.  randomly select images from the addons folder and cut identifiable objects from the images until at least five objects have been captured
#e.  paste the objects from step d onto the original image at random locations
#f.  update the corresponding depth map by sampling the area under the pasted regions of the painting and objects to determine the lowest depth value for each region and applying those same minimum depth values to the entire respective regions of the pasted painting and objects
#g.  save the modified picture, and depth maps in both .hdf5 and .png files in separate folders in the output directory




#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import random
import os
import h5py
import tkinter as tk
import subprocess
import io
from PIL import Image
from tkinter import Tk, filedialog
import shutil
import argparse
import sys
import time
import re

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.utils as vutils
import torch.onnx

from utils import utils
from utils.transformer_net import TransformerNet
from utils.vgg import Vgg16
from shapely.geometry import Polygon

# ---------------------------
# Global Variables
# ---------------------------
rgb_colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 165, 0),   # Orange
    (165, 42, 42),   # Brown
    (128, 0, 128),   # Purple
    (0, 255, 255),   # Cyan
    (255, 192, 203)  # Pink
]
# ---------------------------
# Helper Functions
# ---------------------------

# ------------- FUNCTION: Convert Distance to Depth Map -------------
def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)\
                        .reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)\
                        .reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], axis=2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, ord=2, axis=2) * fltFocal
    return npyDepth

def prompt_for_folder(prompt_message):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=prompt_message)
    if not folder:
        raise ValueError(f"No folder selected for {prompt_message}")
    return folder

def list_files(folder, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def process_manifest(datasource,manifest_path):
    """
    Allows the user to select a text file, reads it line by line, saves each line in an array,
    and splits each line into two values separated by a space.
    """

    # Initialize the manifest array
    manifest = []

    # Open and read the text file line by line
    with open(manifest_path, "r") as file:
        for line in file:
            # Strip any leading/trailing whitespace
            line = line.strip()

            # Split the line into two values delimited by a space
            if " " in line:
                value1, value2 = line.split(" ", 1)
                # Append the values as a row to the array
                manifest.append([value1, value2])
            else:
                print(f"Skipping line (no space found): {line}")

    # File is automatically closed after the `with` block
    print("Finished processing the text file.")
    return manifest

def cartoonize_image(img_path):
    style = np.random.randint(0, 12)
    if style<=5:
        return cv2.imread(img_path)
    elif style<=7:
        img = cv2.imread(img_path)
        """Apply OpenCV stylization to cartoonize an image."""
        sigma_s = 150
        sigma_r = 0.35
        return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        # Call fast_neural_style using the current Python interpreter.
        neural_models = ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'mosaic', 'rain_princess', 'udnie', 'candy']
        neural_model = neural_models[style]
        neural_style_path = r'C:\Users\kenne\Projects\AI\scripts\main\utils\\'
        model = f'{neural_style_path}saved_models/{neural_model}.pth'
        np_img = stylize(model, img_path, 0)
        return np_img

def stylize(model, content_image, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    content_image = utils.load_image(content_image, scale=None)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
        data=output[0]
        np_img = data.clone().clamp(0, 255).numpy()
        np_img = np_img.transpose(1, 2, 0).astype("uint8")

    return np_img
    #utils.save_image(args.output_image, output[0])

def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)

def apply_random_perspective(bkg, region, base_quad, perturbation=.15):
    """
    Given an image region (as a numpy array), compute a random perspective transformation.
    Returns the warped region and the source and destination corner arrays.
    """
    h, w = region.shape[:2]
    bkg_h, bkg_w = bkg.shape[:2]
    # Perturb corners by perturbation.
    dx = w * perturbation
    dy = h * perturbation
    perturbed_quad = []
    for (x, y) in base_quad:
        perturbed_x = np.clip(x + random.uniform(-dx, dx), 0, bkg_w - 1)
        perturbed_y = np.clip(y + random.uniform(-dy, dy), 0, bkg_h - 1)
        perturbed_quad.append([perturbed_x, perturbed_y])
    dest_pts = np.float32(perturbed_quad)
    return dest_pts

def compute_z_values(H, points_2d):
    """
    Given a homography H and a set of 2D points (shape: Nx2),
    convert them to homogeneous coordinates, transform them using H, and return the third
    (homogeneous) coordinate for each point.
    """
    num_pts = points_2d.shape[0]
    ones = np.ones((num_pts, 1), dtype=np.float32)
    points_hom = np.hstack([points_2d, ones])  # shape: (N,3)
    # Multiply by H (note: H is 3x3 and points_hom.T is 3xN)
    transformed = (H @ points_hom.T).T  # shape: (N,3)
    # The third column is our "z-distance" proxy (before normalization)
    z_values = transformed[:, 2]
    return z_values, transformed

def add_border(car, border_width=None, border_color=None):
    """
    Add a border around the image with a specified border width and default brown color.

    Parameters:
        car (np.ndarray): Input image as a NumPy array.
        border_width (int): Width of the border in pixels. Default is 10.
        brown_color (tuple): Border color in BGR format. Default is (42, 42, 165).

    Returns:
        np.ndarray: The image with the added border.
    """
    if border_width is None:
        border_width = np.random.randint(20, 40)
    if border_color is None:
        border_color = random.choice(rgb_colors)
    # cv2.copyMakeBorder adds a border around the image.
    bordered_car = cv2.copyMakeBorder(car, border_width, border_width, border_width, border_width,
                                        borderType=cv2.BORDER_CONSTANT, value=border_color)
    return bordered_car

def paste_car(bkg, region, dest_pts):
    """
    Paste the car_region onto the image
    region_coords: (x, y, w, h) of the region from the left half.
    dest_pts: the quadrilateral (4x2) (in the original image coordinates) where the wall painting will be pasted.
    Returns the modified image.
    """
    bkg_h, bkg_w = bkg.shape[:2]
    # Create an overlay by warping the wall painting region into the destination quadr.
    # Define source points from img2 (its four corners)
    h, w = region.shape[:2]
    src_pts = np.float32(
        [[0, 0],
         [w, 0],
         [w, h],
         [0, h]
         ])
    H = cv2.getPerspectiveTransform(src_pts, dest_pts)
    warped_car = cv2.warpPerspective(region, H, (bkg_w, bkg_h))
    warped_gray = cv2.cvtColor(warped_car, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(warped_gray, 10, 255, cv2.THRESH_BINARY)
    mask_bool = mask.astype(bool)
    bkg_modified = bkg.copy()
    bkg_modified[mask_bool] = warped_car[mask_bool]
    return bkg_modified, warped_car, H

def update_depth_map(depth_map, dest_pts, H_inv, corner_depths):
    """
    For every pixel inside the destination quadrilateral, compute the wall painting's depth via bilinear interpolation
    of the provided corner depths. Then update the depth_map by taking the minimum between the current value and
    the computed wall painting depth.
    """
    xs = dest_pts[:, 0]
    ys = dest_pts[:, 1]
    min_x = int(np.floor(np.min(xs)))
    max_x = int(np.ceil(np.max(xs)))
    min_y = int(np.floor(np.min(ys)))
    max_y = int(np.ceil(np.max(ys)))

    poly = dest_pts.reshape((4, 1, 2)).astype(np.int32)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                pt = np.array([x, y, 1], dtype=np.float32)
                uvw = H_inv.dot(pt)
                if uvw[2] == 0:
                    continue
                u = uvw[0] / uvw[2]
                v = uvw[1] / uvw[2]
                d0, d1, d2, d3 = corner_depths
                computed_depth = (1 - u) * (1 - v) * d0 + u * (1 - v) * d1 + u * v * d2 + (1 - u) * v * d3
                depth_map[y, x] = min(depth_map[y, x], computed_depth)

    return depth_map

# ---------------------------
# Main Processing
# ---------------------------

def process_images_and_depth_maps(datasource,manifest_path,manifest,paintings_folder,output_directory):
    # Define a margin (in meters) that will be added to the minimum background depth.
    # This will affect how pronounced the gradient will be.  A tradeoff is the larger
    # the gradient, the fainter the background will become in a grayscale visual image.
    # Also, there is risk the depth becomes negative or behind the camera, unless
    # the margin is the distance between the minimum depth of the entire original image
    # and the minimum depth of the background behind the pasted quadrilateral.
    margin = 999999 # to be computed later

    # For depth processing, we use a unit-square mapping.
    unit_coords = np.array([[0, 0],
                            [1, 0],
                            [1, 1],
                            [0, 1]], dtype=np.float32)

    # Create output folders
    output_image_folder = os.path.join(output_directory, "output_image_wall")
    output_depth_hdf5_folder = os.path.join(output_directory, "output_depth_hdf5")
    output_depth_png_folder = os.path.join(output_directory, "output_depth_png")
    # Split the file path into directory and filename.
    directory, filename = os.path.split(manifest_path)
    # Split the filename into the base name and extension.
    base, ext = os.path.splitext(filename)
    # Create the new filename by appending the suffix before the extension.
    out_manifest_filename = base + "_out" + ext
    # create output folder for manifest
    out_manifest_dir = os.path.join(output_directory, "output_manifest")
    os.makedirs(out_manifest_dir, exist_ok=True)
    out_manifest_path = os.path.join(out_manifest_dir, out_manifest_filename)

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_depth_hdf5_folder, exist_ok=True)
    os.makedirs(output_depth_png_folder, exist_ok=True)

    for filepath in manifest:
        # Paths for input image and corresponding depth map
        input_image_path = filepath[0]
        input_depth_path = filepath[1]

        base_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
        base_depth_name, depth_ext = os.path.splitext(os.path.basename(input_depth_path))
        # Normalize the path to handle mixed slashes
        normalized_path = os.path.normpath(input_image_path)
        # Split the path into components
        path_parts = normalized_path.split(os.sep)
        if datasource == "hypersim":
            base_image_name = "".join(base_image_name.split(".")[:3])
            base_depth_name = "".join(base_depth_name.split(".")[:3])
            # Extract the desired parts
            scene = path_parts[6].replace("_", "")  # "ai_001_002"
            camera = path_parts[8].replace("_final_preview", "")  # "scene_cam_00"
            # Combine the extracted parts
            prepend_name = f"{scene.replace("_", "")}_{camera.replace("_", "")}_"
            # prepend_name = os.path.normpath(input_image_path).split(os.sep)[6] + "."
        else:
            # Normalize the path to handle mixed slashes
            normalized_path = os.path.normpath(input_image_path)
            # Split the path into components
            path_parts = normalized_path.split(os.sep)
            # Extract the desired parts
            scene = path_parts[7]  # "Scene01"
            angle = path_parts[8].replace("-", "")  # "15degleft"
            camera = path_parts[11].replace("_", "")  # "Camera_1"
            # Combine the extracted parts
            prepend_name = f"{scene}_{angle}_{camera}_"

        # Paths for output image and depth map
        output_image_path = os.path.join(output_image_folder, f"{prepend_name}{base_image_name}_WP.jpg")
        output_depth_hdf5_path = os.path.join(output_depth_hdf5_folder, f"{prepend_name}{base_depth_name}_WP.hdf5")
        output_depth_png_path = os.path.join(output_depth_png_folder, f"{prepend_name}{base_depth_name}_WP.png")

        # Load the image and depth map
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Could not load image: {pic_path}")
            continue
        img_h, img_w = img.shape[:2]
        if datasource == "vkitti":
            depth = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"  Could not load depth image: {depth_path}")
                continue
            depth = depth.astype(np.float32)
        else:
            try:
                with h5py.File(input_depth_path, 'r') as hf:
                    depth = np.array(hf["dataset"], dtype=np.float32)
            except Exception as e:
                print(f"  Error reading depth HDF5: {e}")
                continue

        # Ensure dimensions match
        height, width, _ = img.shape
        if depth.shape != (height, width):
            raise ValueError(f"Depth map dimensions do not match image dimensions for {image_filename}.")
        
        bkg_modified = img.copy()

        painting_files = list_files(paintings_folder, [".jpg", ".png"])
        # === Step 2a: Randomly select a painting and cartoonize it ===
        painting_path = random.choice(painting_files)
        #painting = cv2.imread(painting_path)
        if not (os.path.exists(painting_path) and os.path.isfile(painting_path)):
            print(f"  Warning: could not load painting {painting_path}. Skipping painting step.")
        else:
            unbordered_car = cartoonize_image(painting_path)
            car = add_border(unbordered_car) if np.random.randint(0, 2) else unbordered_car

            # Get dimensions of background and cartoonized image.
            bkg_h, bkg_w = img.shape[:2]
            car_h, car_w = car.shape[:2]

            # ----- Step 2: Resize car so that its width and height are no more than 80% of bkg's dimensions -----
            # Compute a scale factor so that both width and height do not exceed 80% of background's dimensions.
            scale_w = 0.8 * bkg_w / car_w
            scale_h = 0.8 * bkg_h / car_h
            scale = min(scale_w, scale_h)  # minimum of one removed to allow upscaling
            #scale = min(scale_w, scale_h, 1.0)  # Do not enlarge if already small.

            new_w = int(car_w * scale)
            new_h = int(car_h * scale)
            resized_car = cv2.resize(car, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # ----- Step 3: Determine a paste position for the resized image -----
            # For the horizontal position: choose x so that the left edge is between 10% and 70% of bkg's width.
            x_min = int(0.1 * bkg_w)
            x_max = int(0.7 * bkg_w)
            # Ensure that the pasted region fits horizontally.
            if x_max > bkg_w - new_w:
                x_max = bkg_w - new_w
            if x_max < x_min:
                x = x_min
            else:
                x = random.randint(x_min, x_max)

            # For the vertical position: the bottom edge of the pasted image should be between 10% and 20%
            # above the bottom of the background. That is, let offset be between 10% and 20% of bkg's height.
            offset_min = int(0.1 * bkg_h)
            offset_max = int(0.2 * bkg_h)
            offset = random.randint(offset_min, offset_max)
            # Compute y so that the bottom edge of the pasted image is (offset) pixels above the bottom.
            y = bkg_h - new_h - offset
            if y < 0:
                y = 0

            # ----- Step 4: Paste the resized cartoonized image (car) onto the background (bkg) -----
            # If resized_car has an alpha channel, we blend it with the background.
            if resized_car.shape[2] == 4:
                # Split color and alpha.
                color = resized_car[:, :, :3]
                alpha = resized_car[:, :, 3] / 255.0  # normalize alpha to [0,1]
                # Get the region of interest (ROI) in the background.
                roi = bkg[y:y + new_h, x:x + new_w].astype(np.float32)
                # Blend each channel.
                for c in range(3):
                    roi[:, :, c] = alpha * color[:, :, c] + (1 - alpha) * roi[:, :, c]
                # Replace ROI in background with the blended result.
                resized_car = roi.astype(np.uint8)

            #car_bbox = (x, y, new_w, new_h)

            base_quad = np.float32([
                [x, y],
                [x+new_w, y],
                [x+new_w, y + new_h],
                [x, y + new_h]
            ])
            # 2. Apply random perspective to the region.
            dest_pts = apply_random_perspective(img, resized_car, base_quad)
            bkg_modified, warped_car, H = paste_car(bkg_modified, resized_car, dest_pts)
            h_c, w_c = warped_car.shape[:2]
            src_pts = np.float32(
                [[0, 0],
                 [w_c, 0],
                 [w_c, h_c],
                 [0, h_c]
                 ])
            # Save modified image.
            cv2.imwrite(output_image_path, bkg_modified)

            # --- Compute new corner depths based on sampling the entire quadrilateral for the minimum depth ---
            # Create a mask for the destination quadrilateral.
            poly = dest_pts.reshape((4, 1, 2)).astype(np.int32)
            mask_region = np.zeros((bkg_h, bkg_w), dtype=np.uint8)
            cv2.fillPoly(mask_region, [poly], 1)
            if np.count_nonzero(mask_region) == 0:
                # Fallback: sample at corners.
                bg_depths = []
                for (x, y) in dest_pts:
                    x_int = int(round(x))
                    y_int = int(round(y))
                    x_int = np.clip(x_int, 0, bkg_w - 1)
                    y_int = np.clip(y_int, 0, bkg_h - 1)
                    bg_depths.append(depth[y_int, x_int])
                bg_depths = np.array(bg_depths)
            else:
                bg_depths = depth[mask_region == 1]

            min_bg_depth = np.min(bg_depths)
            max_bg_depth = np.max(bg_depths)
            # margin will be calculated as the distance between the minimum depth of the entire original image
            # and the minimum depth of the background behind the pasted quadrilateral.  This is to avoid the
            # adjusted depth of the pasted quadrilateral ending up behind the camera, while maximizing the
            # gradient.
            abs_min_bg_depth = np.min(depth)
            margin = abs_min_bg_depth - min_bg_depth

            # Define the car’s depth range:
            # The car’s "farthest" depth is set to the background's minimum depth.
            # The car’s "farthest" depth is defined as:
            max_car_depth = min_bg_depth

            # Now, assign the maximum car depth (farthest) to the unit-square corner (among the four)
            # that is deepest in the z-direction into the page.
            # Then apply a gradient so that moving away from that corner, the car’s depth decreases
            # (i.e. becomes closer).
            # Compute the z-values for the source corners after applying H.
            z_values, transformed_pts = compute_z_values(H, src_pts)
            # Find the index of the corner with the maximum third coordinate.
            idx_corner_max = int(np.argmax(z_values))

            distances_all = np.linalg.norm(unit_coords - unit_coords[idx_corner_max], axis=1)
            d_max = np.max(distances_all)
            if d_max > 0:
                # For a corner at distance d from the selected (farthest) corner,
                new_corner_depths = max_car_depth + (distances_all / d_max) * (margin)
            else:
                new_corner_depths = np.full(4, max_car_depth, dtype=np.float32)

            # Recompute H_inv (from destination quadr to unit square).
            H_inv = cv2.getPerspectiveTransform(dest_pts, unit_coords)
            updated_depth_data = update_depth_map(depth, dest_pts, H_inv, new_corner_depths)

        # === Step 2f: The depth map was updated in each paste operation using update_depth_in_region.
        # In each case, we sample the region (painting or object) and set its depth to the minimum depth found.

        # Save the updated depth map (distance in meters) to a new HDF5 file.
        if depth_ext == ".hdf5":
            try:
                with h5py.File(output_depth_hdf5_path, 'w') as hf:
                    hf.create_dataset('dataset', data=updated_depth_data)
            except Exception as e:
                print(f"Error writing depth map to {output_depth_hdf5_path}: {e}")
            # Save the updated depth map as a PNG file
            if datasource == "hypersim":
                updated_depth = hypersim_distance_to_depth(updated_depth_data)
                normalized_depth_map = 255 - cv2.normalize(updated_depth, None, 0, 255,
                                                           cv2.NORM_MINMAX)  # dark is 255:maximum depth, light is 0:minimum depth
                # normalized_depth_map = cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX) # dark is 0:minimum depth, light is 255:maximum depth
                depth_map_png = normalized_depth_map.astype(np.uint8)
        else:
            depth_norm = 65535 - cv2.normalize(updated_depth_data, None, 0, 65535,
                                               cv2.NORM_MINMAX)  # dark is 65535:maximum depth, light is 0:minimum depth
            # depth_norm = cv2.normalize(updated_depth_data, None, 0, 65535, cv2.NORM_MINMAX) # dark is 0:minimum depth, light is 65535:maximum depth
            depth_map_png = depth_norm.astype(np.uint16)
        cv2.imwrite(output_depth_png_path, depth_map_png)

        with open(out_manifest_path, "a") as f:
            if depth_ext == ".hdf5":
                f.write(f"{output_image_path} {output_depth_hdf5_path} {output_depth_png_path}\n")
            else:
                f.write(f"{output_image_path} {output_depth_png_path}\n")

        print(f"  Processed {input_image_path}")

def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Process images and depth maps for specific datasets.")
    parser.add_argument("dataset", type=str, nargs="?",
                        help="Specify the dataset type (e.g., 'hypersim' or 'vKITTI').")
    args = parser.parse_args()

    # Handle the 'dataset' argument
    if args.dataset == "hypersim":
        datasource = "hypersim"
    elif args.dataset == "vKITTI":
        datasource = "vkitti"
    else:
        print("Error: Unsupported or missing dataset argument. Use 'hypersim' or 'vKITTI'.")
        sys.exit(1)

    # Hide the root tkinter window
    Tk().withdraw()
    # Let the user select a text file
    manifest_path = filedialog.askopenfilename(title="Select a " + datasource + " Manifest File", filetypes=[("Text Files", "*.txt")])
    if not manifest_path:
        print("No file selected. Exiting.")
        return
    manifest = process_manifest(datasource, manifest_path)

    paintings_folder = prompt_for_folder("Select folder for Indoor Paintings")
    painting_files = list_files(paintings_folder, [".jpg", ".png"])
    if not painting_files:  # or not addon_files:
        raise ValueError("Required folder with images to render Paintings have no files.")

    # Let the user select a directory
    output_directory = filedialog.askdirectory(title="Select a Directory for outputs")

    if not output_directory:
        print("No directory selected. Exiting.")
        sys.exit()

    # Process the images and depth maps
    process_images_and_depth_maps(datasource, manifest_path, manifest, paintings_folder, output_directory)

    print('\nDone adding flat embedded images and processing corresponding depth maps.')

if __name__ == "__main__":
    main()
