#Request:  Please write me some code to automatically take a batch of pictures, randomly add two paintings of large objects to the pictures, give random perspectives to the paintings in #the scene, and then update their associated depth maps considering the paintings are flat and the perspectives of the paintings.

#Request:  Please update your code with the following function (hypersim_distance_to_depth) to convert the depth data (units is distance in meters) in the hdf5 files to depth before saving as png file.  The input #to the function is the original depth data ("dataset"), and the output is a depth map which would be saved as the png file.  The modified depth data (distance would be #saved as the .hdf5 file.  Input depth data ("dataset") is in strictly .hdf5 files.

#Request:  I have run the code.  Incorrect depths are assigned to the paintings pasted onto the original images.  The paintings appear to have depths in the background, rather than the foreground.  Since the paintings are pasted on top of #whatever objects are preexisting on the original images, the paintings should have the lowest depth relative to all objects in the original images.  Please also consider the relative depths of multiple paintings on one image if the #paintings overlap each other.  Please update the code accordingly.

#Response:  Below is the updated code. In this version, when computing the painting’s new depth values, we first sample the current (background) depth at each of the four destination-corner pixels and then subtract a random margin so that the #painting’s depth is always lower (i.e. closer to the camera) than the background. In addition, when updating each pixel inside the painted region, we only override the existing depth if the computed painting depth is lower than what is #already there. This way, if multiple paintings overlap, the one with the lowest (closest) depth remains visible.

#Request:  Your code still assigns higher depth values to paintings that are clearly in front of background objects in the original scenes.  Please update code.  After you sample the background region of the pasted painting, please use the lowest #depth value in the sampling, apply this lowest depth value to the corner of the painting that is closest to the camera, and then apply a gradient to the depth of the painting starting with this corner based on the perspective of the #painting.

#Response:  Below is the updated code. In this version, when a painting is pasted, the code samples the background depth at the four destination‐quadrilateral corners. Then it finds the minimum (closest) depth among those samples and assigns that #value to the “closest” painting corner (determined by the corresponding unit‐square coordinate). For the other three corners, a gradient is applied so that the farther corners are assigned progressively higher (i.e. further away) depth #values. In our example the gradient is computed in the unit‐square domain: we assign each corner a depth according to...

#Request:  Please update code to determine the min_depth.  In your code, you sample the background depth at only the four corners.  Please update code to sample the background depth within the entire quadrilateral background region covered by the #pasted painting.

#Request:  Please update code to apply the min_depth to the corner that is farthest in depth, and then apply the gradient in reverse - meaning the painting would always have a gradient of decreasing or the same depth emanating from the corner that is deepest. 
#Wrong from ChatGPT:  Below is the updated code. In this version, rather than assigning the painting’s “closest‐to‑camera” (minimum) depth to the pixel with the lowest background depth, we now sample the entire quadrilateral region of the background and determine the pixel with the highest (i.e. farthest) depth. We then define the painting’s minimum (closest) depth as that maximum background depth minus a fixed margin. That value is applied to the corner of the painting (in unit‐square coordinates) that is closest to the farthest background pixel. Then, a gradient is applied in “reverse” – meaning that moving away from that deepest corner, the painting’s depth increases (i.e. becomes farther) up to nearly the original background value. (In other words, the painting will always have a gradient of increasing depth emanating from the corner that was farthest in depth in the background.) You may adjust the margin parameter as needed.
#Corrected mannually:  Please update code in accordance with:  In this version, assign the painting’s “closest‐to‑camera” (minimum) depth to the pixel with the lowest background depth by sampling the entire quadrilateral region of the background and determine the pixel with the lowest (i.e. closest) depth. We then define the painting’s maximum (farthest) depth as that minimum background depth minus a fixed margin. That value is applied to the corner of the painting (in unit‐square coordinates) that is closest to the farthest background pixel. Then, a gradient is applied – meaning that moving away from that deepest corner, the painting’s depth decreases (i.e. becomes closer) up to be equal to or less than the background value. (In other words, the painting will always have a gradient of decreasing depth emanating from the corner that was farthest in depth in the background.)  

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
# Helper functions
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

def prompt_for_manifest():
    """Prompt the user to select a manifest file."""
    root = tk.Tk()
    root.withdraw()
    manifest_path = filedialog.askopenfilename(title="Select Manifest File",
                                               filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if not manifest_path:
        raise ValueError("No manifest file selected.")
    return manifest_path

def prompt_for_output_directory():
    """Prompt the user to select an output directory."""
    root = tk.Tk()
    root.withdraw()
    out_dir = filedialog.askdirectory(title="Select Output Directory")
    if not out_dir:
        raise ValueError("No output directory selected.")
    return out_dir

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

def random_dest_quad(img_shape, painting_shape, scale_range=(0.2, 0.5), perturbation=0.1):
    """
    Create a random destination quadrilateral inside an image.
    """
    img_h, img_w = img_shape[:2]
    p_h, p_w = painting_shape

    scale = random.uniform(*scale_range)/.95
    max_x = -1
    while (max_x <= 0 or max_y <= 0) and scale >= min(scale_range):
        scale = scale*.95
        target_w = int(img_w * scale)
        aspect = p_h / p_w
        target_h = int(target_w * aspect)
        max_x = img_w - target_w - 1
        max_y = img_h - target_h - 1

    if scale < min(scale_range):
        raise ValueError("Image too small for painting size")

    x0 = random.randint(0, max_x)
    y0 = random.randint(0, max_y)

    base_quad = np.float32([
        [x0, y0],
        [x0 + target_w, y0],
        [x0 + target_w, y0 + target_h],
        [x0, y0 + target_h]
    ])
    
    dx = target_w * perturbation
    dy = target_h * perturbation
    perturbed_quad = []
    for (x, y) in base_quad:
        perturbed_x = np.clip(x + random.uniform(-dx, dx), 0, img_w - 1)
        perturbed_y = np.clip(y + random.uniform(-dy, dy), 0, img_h - 1)
        perturbed_quad.append([perturbed_x, perturbed_y])
    return np.float32(perturbed_quad)

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

def add_border(painting, border_width=None, border_color=None):
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
        border_width = np.random.randint(15, 30)
    if border_color is None:
        border_color = random.choice(rgb_colors)
    # cv2.copyMakeBorder adds a border around the image.
    bordered_painting = cv2.copyMakeBorder(painting, border_width, border_width, border_width, border_width,
                                        borderType=cv2.BORDER_CONSTANT, value=border_color)
    return bordered_painting

def overlay_image(background, overlay, mask):
    """
    Overlay the overlay image on the background where mask is non-zero.
    """
    mask_bool = mask.astype(bool)
    result = background.copy()
    result[mask_bool] = overlay[mask_bool]
    return result

def update_depth_map(depth_map, dest_pts, H_inv, corner_depths):
    """
    For every pixel inside the destination quadrilateral, compute the painting's depth via bilinear interpolation
    of the provided corner depths. Then update the depth_map by taking the minimum between the current value and
    the computed painting depth.
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

def process_images_and_depth_maps(datasource,manifest_path,manifest,paintings_dir,output_directory):
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

    painting_files = sorted(glob.glob(os.path.join(paintings_dir, '*.jpg')))

    # Create output folders
    output_image_folder = os.path.join(output_directory, "output_image_flat")
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
        if datasource == "hypersim":
            base_image_name = ".".join(base_image_name.split(".")[:3])
            base_depth_name = ".".join(base_depth_name.split(".")[:3])
            prepend_name=""
            #prepend_name = os.path.normpath(input_image_path).split(os.sep)[6] + "."
        else:
            # Normalize the path to handle mixed slashes
            normalized_path = os.path.normpath(input_image_path)
            # Split the path into components
            path_parts = normalized_path.split(os.sep)
            # Extract the desired parts
            scene = path_parts[7]  # "Scene01"
            angle = path_parts[8].replace("-", "")  # "15degleft"
            # Combine the extracted parts
            prepend_name = f"{scene}_{angle}_"

        # Paths for output image and depth map
        output_image_path = os.path.join(output_image_folder, f"{prepend_name}{base_image_name}_FP.jpg")
        output_depth_hdf5_path = os.path.join(output_depth_hdf5_folder, f"{prepend_name}{base_depth_name}_FP.hdf5")
        output_depth_png_path = os.path.join(output_depth_png_folder, f"{prepend_name}{base_depth_name}_FP.png")

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
        else:
            try:
                with h5py.File(input_depth_path, 'r') as hf:
                    depth = np.array(hf["dataset"], dtype=np.float32)
            except Exception as e:
                print(f"  Error reading depth HDF5: {e}")
                continue

        # Ensure dimensions match
        if depth.shape != (img_h, img_w):
            raise ValueError(f"Depth map dimensions do not match image dimensions for {image_filename}.")

        updated_depth_data = depth.copy()
    
        # Process two paintings per image.
        for i in range(2):
            painting_path = random.choice(painting_files)
            raw_painting = cv2.imread(painting_path)
            if raw_painting is None:
                continue
            painting = add_border(raw_painting) if np.random.randint(0, 2) else raw_painting
            p_h, p_w = painting.shape[:2]
            # Scale down if necessary.
            if p_w >= img_w or p_h >= img_h:
                scale_factor = min(img_w / p_w, img_h / p_h)
                painting = cv2.resize(painting, (int(p_w * scale_factor), int(p_h * scale_factor)),
                                        interpolation=cv2.INTER_AREA)
                p_h, p_w = painting.shape[:2]

            src_pts = np.float32([[0, 0],
                                  [p_w, 0],
                                  [p_w, p_h],
                                  [0, p_h]])
            dest_pts = random_dest_quad(img.shape, (p_h, p_w), scale_range=(0.2, 0.5), perturbation=0.15)
            H = cv2.getPerspectiveTransform(src_pts, dest_pts)
            warped_painting = cv2.warpPerspective(painting, H, (img_w, img_h))
            warped_gray = cv2.cvtColor(warped_painting, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(warped_gray, 10, 255, cv2.THRESH_BINARY)
            img = overlay_image(img, warped_painting, mask)

            # --- Compute new corner depths based on sampling the entire quadrilateral for the minimum depth ---
            # Create a mask for the destination quadrilateral.
            poly = dest_pts.reshape((4, 1, 2)).astype(np.int32)
            mask_region = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask_region, [poly], 1)
            if np.count_nonzero(mask_region) == 0:
                # Fallback: sample at corners.
                bg_depths = []
                for (x, y) in dest_pts:
                    x_int = int(round(x))
                    y_int = int(round(y))
                    x_int = np.clip(x_int, 0, img_w - 1)
                    y_int = np.clip(y_int, 0, img_h - 1)
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

            # Define the painting’s depth range:
            # The painting’s "farthest" depth is set to the background's minimum depth.
            # The painting’s "farthest" depth is defined as:
            max_paint_depth = min_bg_depth

            # Now, assign the maximum painting depth (farthest) to the unit-square corner (among the four)
            # that is deepest in the z-direction into the page.
            # Then apply a gradient so that moving away from that corner, the painting’s depth decreases
            # (i.e. becomes closer).
            # Compute the z-values for the source corners after applying H.
            z_values, transformed_pts = compute_z_values(H, src_pts)
            # Find the index of the corner with the maximum third coordinate.
            idx_corner_max = int(np.argmax(z_values))

            distances_all = np.linalg.norm(unit_coords - unit_coords[idx_corner_max], axis=1)
            d_max = np.max(distances_all)
            if d_max > 0:
                # For a corner at distance d from the selected (farthest) corner,
                new_corner_depths = max_paint_depth + (distances_all / d_max) * (margin)
            else:
                new_corner_depths = np.full(4, max_paint_depth, dtype=np.float32)

            # Recompute H_inv (from destination quadr to unit square).
            H_inv = cv2.getPerspectiveTransform(dest_pts, unit_coords)
            updated_depth_data = update_depth_map(depth, dest_pts, H_inv, new_corner_depths)

        # Save the augmented image.
        cv2.imwrite(output_image_path, img)
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
                normalized_depth_map = cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_map_png = normalized_depth_map.astype(np.uint8)
        else:
            depth_norm = cv2.normalize(updated_depth_data, None, 0, 65535, cv2.NORM_MINMAX)
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

    # Let the user select a folder with images for paintings
    print("Select the folder containing the painting images:")
    paintings_dir = filedialog.askdirectory(title="Select Painting Images Folder")
    if not paintings_dir:
        raise ValueError("No folder selected for painting images.")
    painting_files = sorted(glob.glob(os.path.join(paintings_dir, '*.jpg')))
    if not painting_files:
        raise ValueError("No painting images found in the selected folder.")

    # Let the user select an output directory
    output_directory = filedialog.askdirectory(title="Select a Directory for outputs")
    if not output_directory:
        print("No directory selected. Exiting.")
        sys.exit()

    # Process the images and depth maps
    process_images_and_depth_maps(datasource, manifest_path, manifest, paintings_dir, output_directory)

    print('\nDone adding flat embedded images and processing corresponding depth maps.')


if __name__ == "__main__":
    main()
