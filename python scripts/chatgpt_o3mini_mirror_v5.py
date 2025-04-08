#Request:  Please provide a python script to perform the following:
#1.  prompt user to select a manifest file with file paths of pictures (e.g. png or jpg files) and file paths of grayscale images (.hdf5 or .png files), delimited by a space in each line
#3.  prompt user to select an output directory
#4.  Processing each line in the manifest file, for each picture, perform:
#a.  copy a random portion of the left half, between 50%-80% of the size of the left half
#b.  apply random perspective to the copied region
#c.  flip the copied image to simulate a mirror image
#d.  paste the mirror image onto the right side of the original image such that it corresponds to a mirror reflection of the left side
#e.  save the modified picture with the same name with suffix "_MP" in a new output folder in the output directory
#f.  open the corresponding grayscale image assuming the depth data is labeled as "dataset"
#g.  within the grayscale image, sample the same quadrilateral region of the location of the pasted mirror reflection, and determine the lowest depth value (min_depth)
#h.  apply this min-depth to the corner of the mirror reflection that is deepest according to the perspective of the mirror reflection
#i.  apply a gradient to the depth values of the rest of the mirror reflection such that the corner that is deepest has the lowest depth value, and the corner that is closest has an equal or lower depth value
#j.  save the modified depth map with the same name with suffix "_MP" in a new folder (output_hdf5 if .hdf5 file or output_png if .png file) in the output directory
#k.  if the original depth map is a .hdf5 file, then save the depth map also as a .png file with the same name and suffix "_MP" in a new folder (output_png) in the output #directory
#l.  if the original depth map is a .png or .jpg file, then save the depth map also as a .hdf5 file with the same name and suffix "_MP" in a new folder (output_hdf5) in the output directory

#Request:  Please update code to modify only the mirror reflection quadrilateral region of the depth map.  In other words, determine the matching quadrilateral region on the depth map to the original image where the mirror reflection is pasted, and modify the depth data only for that region of interest.

#Response:  Below is the updated Python script. In this version, when updating the depth map, only the mirror‐reflection quadrilateral region (i.e. the region where the mirror was pasted) is modified. In the code below, we first create a copy of the original depth map and then update only those pixels that fall inside the mirror quadrilateral (using the mask derived from that quadrilateral). Outside this region the depth values remain unchanged.

#Request:  Please analyze this script (chatgpt_o3mini_flat.py) which you provided yesterday with my modifications, and use it as an example of how the script for mirror reflections need to be modified to correctly sample the background depth map of only the quadrilateral region of the mirror reflection and to update depth map of only the quadrilateral region.

#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import random
import argparse
import sys
import os
import h5py
from tkinter import Tk, filedialog
import shutil

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

def random_half_region(img_shape, left_or_right):
    """
    Return a random rectangular region from the left or right half of the image.
    The region size is chosen between 50%-80% of the left half dimensions.
    Returns (x, y, w, h).
    """
    h, w = img_shape[:2]
    # Random region width and height (as fraction of half and full height)
    region_w = int(random.uniform(0.75, 0.9) * w//2) # 021625 - changed lower bound from 0.5 to 0.15
    region_h = int(random.uniform(0.75, 0.9) * h) # 021625 - changed lower bound from 0.5 to 0.15
    # Random top-left such that region fits within left half
    x = random.randint(0, w//2 - region_w) if left_or_right==0 else random.randint(w//2, w - region_w)
    y = random.randint(0, h - region_h)
    return x, y, region_w, region_h

def apply_random_perspective(img_shape,region,base_quad,left_or_right):
    """
    Given an image region (as a numpy array), compute a random perspective transformation.
    Returns the warped region and the source and destination corner arrays.
    """
    img_h, img_w = img_shape[:2]
    h, w = region.shape[:2]
    # Perturb corners by up to 10% of width/height.
    dx = 0.1 * w
    dy = 0.1 * h
    # start the mirror with a slight angle to face the center of the scene
    if left_or_right==0:
        base_quad[1,0] -= 3*dx
        base_quad[2,0] -= 3*dx
        base_quad[1,1] = np.clip(base_quad[1, 1] - 3*dy, 0, img_h-1)
        base_quad[2,1] = np.clip(base_quad[2, 1] + 3*dy, 0, img_h-1)
    else:
        base_quad[0,0] += 3*dx
        base_quad[3,0] += 3*dx
        base_quad[0,1] = np.clip(base_quad[1, 1] - 3*dy, 0, img_h-1)
        base_quad[3,1] = np.clip(base_quad[2, 1] + 3*dy, 0, img_h-1)
    dest_pts = []
    x=base_quad[0,0]
    y=base_quad[0,1]
    perturbed_x = x#np.clip(x + random.uniform(-dx, dx), 0, img_w - 1)
    perturbed_y = y#np.clip(y + random.uniform(-dy, dy), 0, img_h - 1)
    dest_pts.append([perturbed_x, perturbed_y])
    dest_pts.append([base_quad[1,0],base_quad[1,1]])
    dest_pts.append([base_quad[2,0],base_quad[2,1]])
    x=base_quad[3,0]
    y=base_quad[3,1]
    perturbed_x = x#np.clip(x + random.uniform(-dx, dx), 0, img_w - 1)
    perturbed_y = y#np.clip(y + random.uniform(-dy, dy), 0, img_h - 1)
    dest_pts.append([perturbed_x, perturbed_y])
    dest_pts = np.float32(dest_pts)
    return dest_pts

def mirror_region(region):
    """Flip the region horizontally."""
    return cv2.flip(region, 1)

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

def add_border(mirror, border_width=None, border_color=None):
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
        border_width = np.random.randint(10, 25)
    if border_color is None:
        border_color = random.choice(rgb_colors)
    # cv2.copyMakeBorder adds a border around the image.
    bordered_mirror = cv2.copyMakeBorder(mirror, border_width, border_width, border_width, border_width,
                                        borderType=cv2.BORDER_CONSTANT, value=border_color)
    return bordered_mirror

def paste_mirror(image,region,dest_pts):
    """
    Paste the mirror_region onto the image on the right side.
    region_coords: (x, y, w, h) of the region from the left half.
    dest_pts: the quadrilateral (4x2) (in the original image coordinates) where the mirror will be pasted.
    Returns the modified image.
    """

    img_h, img_w = image.shape[:2]
    # Create an overlay by warping the mirror region into the destination quadr.
    # Define source points from img2 (its four corners)
    h, w = region.shape[:2]
    src_pts = np.float32(
        [[0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    H = cv2.getPerspectiveTransform(src_pts, dest_pts)
    warped_mirror = cv2.warpPerspective(region, H, (img_w, img_h))
    warped_gray = cv2.cvtColor(warped_mirror, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(warped_gray, 10, 255, cv2.THRESH_BINARY)
    mask_bool = mask.astype(bool)
    img_modified = image.copy()
    img_modified[mask_bool] = warped_mirror[mask_bool]
    return img_modified, warped_mirror, H

def update_depth_map(depth_map, dest_pts, H_inv, corner_depths):
    """
    For every pixel inside the destination quadrilateral, compute the mirror's depth via bilinear interpolation
    of the provided corner depths. Then update the depth_map by taking the minimum between the current value and
    the computed mirror depth.
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
# Main processing
# ---------------------------

def process_images_and_depth_maps(datasource,manifest_path,manifest,output_directory):
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
    output_image_folder = os.path.join(output_directory, "output_image_mirror")
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
        output_image_path = os.path.join(output_image_folder, f"{prepend_name}{base_image_name}_MP.jpg")
        output_depth_hdf5_path = os.path.join(output_depth_hdf5_folder, f"{prepend_name}{base_depth_name}_MP.hdf5")
        output_depth_png_path = os.path.join(output_depth_png_folder, f"{prepend_name}{base_depth_name}_MP.png")

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

        # Handle NaN and Inf
        depth = np.nan_to_num(depth, nan=0.2, posinf=15, neginf=0.2)

        # Ensure dimensions match
        height, width, _ = img.shape
        if depth.shape != (height, width):
            raise ValueError(f"Depth map dimensions do not match image dimensions for {image_filename}.")
        
        # --- Process the picture ---
        # 1. Select a random region from the left half.
        left_or_right = random.randint(0, 1)
        x, y, w_region, h_region = random_half_region(img.shape, left_or_right)
        region = img[y:y+h_region, x:x+w_region].copy()
        # 3. Flip the mirror region horizontally.
        flipped_mirror = mirror_region(region)
        mirror = add_border(flipped_mirror) if np.random.randint(0, 2) else flipped_mirror
        # Place the mirror on the opposite half as the symmetric counterpart.
        base_quad = np.float32([
            [img_w - (x + w_region), y],
            [img_w - x, y],
            [img_w - x, y + h_region],
            [img_w - (x + w_region), y + h_region]
        ])
        # 2. Apply random perspective to the region.
        dest_pts = apply_random_perspective(img.shape,mirror,base_quad,left_or_right)
        img_modified, warped_mirror, H = paste_mirror(img, mirror, dest_pts)
        h_m, w_m = warped_mirror.shape[:2]
        src_pts = np.float32(
            [[0, 0],
            [w_m, 0],
            [w_m, h_m],
            [0, h_m]
        ])
        cv2.imwrite(output_image_path, img_modified)
        
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

        # Define the mirror’s depth range:
        # The mirror’s "farthest" depth is set to the background's minimum depth.
        # The mirror’s "farthest" depth is defined as:
        max_mirror_depth = min_bg_depth
        
        # Now, assign the maximum mirror depth (farthest) to the unit-square corner (among the four)
        # that is deepest in the z-direction into the page.
        # Then apply a gradient so that moving away from that corner, the mirror’s depth decreases
        # (i.e. becomes closer).
        # Compute the z-values for the source corners after applying H.
        z_values, transformed_pts = compute_z_values(H, src_pts)
        # Find the index of the corner with the maximum third coordinate.
        idx_corner_max = int(np.argmax(z_values))

        distances_all = np.linalg.norm(unit_coords - unit_coords[idx_corner_max], axis=1)
        d_max = np.max(distances_all)
        if d_max > 0:
            # For a corner at distance d from the selected (farthest) corner,
            new_corner_depths = max_mirror_depth + (distances_all / d_max) * (margin)
        else:
            new_corner_depths = np.full(4, max_mirror_depth, dtype=np.float32)
        
        # Recompute H_inv (from destination quadr to unit square).
        H_inv = cv2.getPerspectiveTransform(dest_pts, unit_coords)
        updated_depth_data = update_depth_map(depth, dest_pts, H_inv, new_corner_depths)
        
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
                normalized_depth_map = 255-cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX) # dark is 255:maximum depth, light is 0:minimum depth
                #normalized_depth_map = cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX) # dark is 0:minimum depth, light is 255:maximum depth
                depth_map_png = normalized_depth_map.astype(np.uint8)
        else:
            depth_norm = 65535-cv2.normalize(updated_depth_data, None, 0, 65535, cv2.NORM_MINMAX) # dark is 65535:maximum depth, light is 0:minimum depth
            #depth_norm = cv2.normalize(updated_depth_data, None, 0, 65535, cv2.NORM_MINMAX) # dark is 0:minimum depth, light is 65535:maximum depth
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
    # Let the user select a directory
    output_directory = filedialog.askdirectory(title="Select a Directory for outputs")

    if not output_directory:
        print("No directory selected. Exiting.")
        sys.exit()

    # Process the images and depth maps
    process_images_and_depth_maps(datasource, manifest_path, manifest, output_directory)

    print('\nDone adding flat embedded images and processing corresponding depth maps.')


if __name__ == "__main__":
    main()