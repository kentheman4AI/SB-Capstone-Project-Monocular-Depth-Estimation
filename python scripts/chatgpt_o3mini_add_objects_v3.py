# pip install opencv-python opencv-contrib-python rembg Pillow numpy h5py

# !/usr/bin/env python3
import cv2
import numpy as np
import os
import h5py
import glob
import random
import tkinter as tk
from tkinter import filedialog
from rembg import remove
from PIL import Image
import io


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

def prompt_for_file(prompt_message):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt_message)
    if not file_path:
        raise ValueError(f"No file selected for {prompt_message}")
    return file_path


def prompt_for_folder(prompt_message):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=prompt_message)
    if not folder:
        raise ValueError(f"No folder selected for {prompt_message}")
    return folder


def list_files(folder, extensions):
    """
    List and return files in the folder that have one of the given extensions.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def remove_background_from_file(addon_path):
    """
    Uses rembg to remove the background from an addon image.
    Returns a tuple (cv2_img, mask) where:
      - cv2_img is the resulting image as a BGRA NumPy array,
      - mask is a binary mask (values 0 or 255) created from the alpha channel.
    """
    # Read the addon image as bytes.
    with open(addon_path, 'rb') as f:
        input_bytes = f.read()

    # Pass the raw bytes to rembg.remove.
    output_bytes = remove(input_bytes)

    # Convert the output bytes to a PIL image.
    output_pil = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Convert the PIL image to a NumPy array.
    np_img = np.array(output_pil)

    # Create a binary mask from the alpha channel.
    # We'll threshold the alpha channel to create a binary mask.
    alpha = np_img[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255  # pixels with alpha > 10 become 255 (object)

    # Convert the image to BGRA (for OpenCV).
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)

    return cv_img, mask


def enlarge_image(image, scale_factor=3.0):
    """
    Enlarge the image by the given scale factor.
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    enlarged = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return enlarged


def paste_region_near_bottom(base_img, region_img, bottom_offset_range=(0, 0.1)):
    """
    Paste region_img onto base_img so that its bottom edge is at a random vertical offset
    (as a fraction of base image height) from the bottom.
    If region_img is larger than base_img in any dimension, it is scaled down to fit.
    Horizontal position is chosen at random.
    Returns the modified image and the bounding box (x, y, w, h) of the pasted region.
    """
    base_h, base_w = base_img.shape[:2]
    reg_h, reg_w = region_img.shape[:2]

    # Scale down if necessary.
    if reg_w >= base_w or reg_h >= base_h:
        scale_factor = min(base_w / reg_w, base_h / reg_h)
        region_img = cv2.resize(region_img, (int(reg_w * scale_factor), int(reg_h * scale_factor)),
                                interpolation=cv2.INTER_AREA)
        reg_h, reg_w = region_img.shape[:2]

    # If, for some reason, reg_h or reg_w is 0, return the base image without changes.
    if reg_h == 0 or reg_w == 0:
        print("Warning: Pasted region has zero size.")
        return base_img, (0, 0, 0, 0)

    x = random.randint(0, max((base_w - reg_w),1))
    offset_fraction = random.uniform(*bottom_offset_range)
    y = int(base_h - reg_h - (offset_fraction * base_h))

    out_img = base_img.copy()
    if region_img.shape[2] == 4:
        color = region_img[:, :, :3]
        alpha = region_img[:, :, 3] / 255.0
        for c in range(3):
            # Ensure the slice has non-zero size
            roi = out_img[y:y + reg_h, x:x + reg_w, c]
            if roi.size == 0:
                continue
            out_img[y:y + reg_h, x:x + reg_w, c] = (alpha * color[:, :, c] +
                                                    (1 - alpha) * roi)
        if roi.shape != alpha.shape:
            print('Region to paste is not pastable')
            return base_img, (0, 0, 0, 0)
    else:
        out_img[y:y + reg_h, x:x + reg_w] = region_img
    return out_img, (x, y, reg_w, reg_h)


def get_mask_dimensions(mask):
    """
    Compute the dimension of the region where the mask is 1 for width and height.

    Parameters:
      mask (np.ndarray): A 2D binary NumPy array (e.g. values 0 and 1, or 0 and 255).
      dim 0, compute the width (span of column indices).
      dim 1, compute the height (span of row indices).

    Returns:
      int: The computed dimension (in pixels) of the region where mask==1.
           Returns 0 if no pixel equals 1.
    """
    # Find the indices where the mask equals 1.
    indices = np.argwhere(mask == 1)

    # If no pixels are 1, return 0.
    if indices.size == 0:
        return 0,0

    # Select the relevant axis:
    # For width, we use column indices (index 1);
    # for height, we use row indices (index 0).
    relevant_w = indices[:, 1]
    relevant_h = indices[:, 0]

    # The dimension is the difference between the maximum and minimum index plus one.
    return int(relevant_w.max() - relevant_w.min() + 1),int(relevant_h.max() - relevant_h.min() + 1)


def is_valid_bbox(depth, bbox):
    """
    Check if the bounding box is a valid slice of the depth map.

    Parameters:
      depth : NumPy array representing the depth map (or any image).
      bbox  : Tuple (x, y, w, h) where:
              x, y are the top-left coordinates,
              w is the width, and
              h is the height.

    Returns:
      True if the bounding box is valid; False otherwise.
    """
    x, y, w, h = bbox
    H, W = depth.shape[:2]
    # Check that width and height are positive.
    if w <= 0 or h <= 0:
        return False
    # Check that the top-left corner is within the image.
    if x < 0 or y < 0:
        return False
    # Check that the entire region lies within the image boundaries.
    if x + w > W or y + h > H:
        return False
    return True


def update_depth_using_mask(depth, bbox, mask):
    """
    Given a depth map (float32), a bounding box (x, y, w, h), and a binary mask (0 or 255)
    corresponding to an object region, resize the mask (if necessary) to match the bbox,
    then sample the depth values in the masked region and update that region in the depth map
    to the minimum depth value.
    """
    x, y, w, h = bbox
    # Resize the mask to bbox size if needed.
    if mask.shape[0] != h or mask.shape[1] != w:
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = mask
    region = depth[y:y + h, x:x + w]

    object_depths = region[resized_mask == 255]
    if object_depths.size > 0:
        min_val = np.min(object_depths)
        region[resized_mask == 255] = min_val
    depth[y:y + h, x:x + w] = region
    return depth


# ---------------------------
# Main Processing
# ---------------------------
def main():
    # 1. Prompt for a manifest file.
    manifest_path = prompt_for_file(
        "Select manifest file (each line: background_path depth_path, separated by a space)")
    manifest_pairs = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            bg_path = parts[0].strip()
            depth_path = parts[1].strip()
            manifest_pairs.append((bg_path, depth_path))
    if not manifest_pairs:
        raise ValueError("No valid entries in manifest file.")

    # 2. Prompt for an addons folder.
    addons_folder = prompt_for_folder("Select addons folder")
    addon_files = list_files(addons_folder, [".jpg", ".png"])
    if not addon_files:
        raise ValueError("No addon images found in the addons folder.")

    # 3. Prompt for an output directory.
    output_directory = prompt_for_folder("Select output directory")
    # Create output folders
    output_image_folder = os.path.join(output_directory, "output_final_image")
    output_depth_hdf5_folder = os.path.join(output_directory, "output_final_depth_hdf5")
    output_depth_png_folder = os.path.join(output_directory, "output_final_depth_png")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_depth_hdf5_folder, exist_ok=True)
    os.makedirs(output_depth_png_folder, exist_ok=True)

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

    # 4. Process each background image from the manifest.
    for bg_path, depth_path in manifest_pairs:
        print(f"Processing background: {bg_path}")
        # Paths for output image and depth map
        base_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_ext = os.path.splitext(bg_path)[1]
        base_depth_name, depth_ext = os.path.splitext(os.path.basename(depth_path))
        out_img_path = os.path.join(output_image_folder, base_name + "_OP" + bg_ext)
        output_depth_hdf5_path = os.path.join(output_depth_hdf5_folder, base_depth_name + "_OP.hdf5")
        output_depth_png_path = os.path.join(output_depth_png_folder, base_depth_name + "_OP.png")
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            print(f"  Warning: could not load background image {bg_path}. Skipping.")
            continue

        # Load corresponding depth map.
        depth_ext = os.path.splitext(depth_path)[1].lower()
        if depth_ext == ".png":
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                print(f"  Warning: could not load depth map {depth_path}. Skipping.")
                continue
            depth = depth.astype(np.float32)
        elif depth_ext == ".hdf5":
            try:
                with h5py.File(depth_path, 'r') as hf:
                    depth = np.array(hf["dataset"], dtype=np.float32)
            except Exception as e:
                print(f"  Warning: could not load depth map {depth_path}: {e}. Skipping.")
                continue
        else:
            print(f"  Warning: unsupported depth map format {depth_path}. Skipping.")
            continue

        modified_bg = bg_img.copy()
        modified_depth = depth.copy()
        base_h, base_w = modified_bg.shape[:2]
        bottom_offset_range = (0.0, 0.1)

        # 4a. Randomly process addon images until three objects have been extracted.
        extracted_objects = []
        used_addons = set()
        attempts = 0
        max_attempts = 30  # Avoid infinite loops.
        while len(extracted_objects) < 3 and attempts < max_attempts:
            addon_path = random.choice(addon_files)
            # Skip if we've already used this addon.
            if addon_path in used_addons:
                attempts += 1
                continue
            used_addons.add(addon_path)
            try:
                obj_img, obj_mask = remove_background_from_file(addon_path)
                if obj_img is not None:
                    extracted_objects.append((obj_img, obj_mask))
            except Exception as e:
                print(f"  Warning: failed to extract object from {addon_path}: {e}")
            attempts += 1

        if len(extracted_objects) < 5:
            print(f"  Warning: Only {len(extracted_objects)} objects extracted for {bg_path}.")
        else:
            extracted_objects = extracted_objects[:5]

        # 4b. Record bounding boxes of pasted objects.
        pasted_regions = []

        # 4c. Paste the five extracted objects randomly onto the background image.
        # 4d. Record their bounding boxes.
        # 4e & 4f. For each pasted object, sample the corresponding region in the depth map
        # and update that region with the minimum depth (only where the object is present, using the mask).
        for (obj_img, obj_mask) in extracted_objects:
            reg_h, reg_w = obj_img.shape[:2]
            # Scale down if necessary.
            scale_factor = min(2.0, base_w / reg_w, base_h / reg_h) # initial scale factor (enlarge by 200%)
            min_scale_factor = 0.5  # do not scale below 50% of the original enlargement factor
            while True:
                # Enlarge the object and mask by the current scale factor.
                enlarged_obj = enlarge_image(obj_img, scale_factor=scale_factor)
                enlarged_mask = enlarge_image(obj_mask, scale_factor=scale_factor)
                reg_h, reg_w = enlarged_obj.shape[:2]
                # Randomly choose a horizontal position that fits.
                if (reg_w < base_w) and (reg_h < base_h):
                    x = random.randint(0, base_w - reg_w)
                    # Choose a random vertical offset (as a fraction of base height) from the bottom.
                    offset_fraction = random.uniform(*bottom_offset_range)
                    # Compute y so that the bottom edge of the object is offset_fraction * base_h above the bottom.
                    y = int(base_h - reg_h - (offset_fraction * base_h))
                    bbox = (x, y, reg_w, reg_h)
                    # Check if the computed bounding box is valid.
                    if is_valid_bbox(modified_depth, bbox):
                        # Valid bounding box found.
                        break
                    else:
                        # Reduce the scale factor and try again.
                        scale_factor *= 0.9
                        if scale_factor < min_scale_factor:
                            print("Warning: minimum scale factor reached; using current values.")
                            break
                else:
                    # Reduce the scale factor and try again.
                    scale_factor *= 0.9
                    if scale_factor < min_scale_factor:
                        print("Warning: minimum scale factor reached; using current values.")
                        break

            # Enlarge the object by 300%.
            enlarged_obj = enlarge_image(obj_img, scale_factor)
            enlarged_mask = enlarge_image(obj_mask, scale_factor)
            # Paste the object onto the background near the bottom (vertical offset between 0%-10% from bottom).
            modified_bg, bbox = paste_region_near_bottom(modified_bg, enlarged_obj, bottom_offset_range)
            pasted_regions.append(bbox)
            # # Update the depth map for this region using the mask.
            # modified_depth = update_depth_using_mask(modified_depth, bbox, enlarged_mask)
            if is_valid_bbox(modified_depth, bbox):
                pasted_regions.append(bbox)
                # Update the depth map for this region using the mask.
                modified_depth = update_depth_using_mask(modified_depth, bbox, enlarged_mask)
            else:
                print("The bounding box is not valid.  Depth map will not be modified for this object.")

        # 4g. Save the modified background image.
        cv2.imwrite(out_img_path, modified_bg)

        # Save the updated depth map (distance in meters) to a new HDF5 file.
        if depth_ext == ".hdf5":
            try:
                with h5py.File(output_depth_hdf5_path, 'w') as hf:
                    hf.create_dataset('dataset', data=modified_depth)
            except Exception as e:
                print(f"Error writing depth map to {output_depth_hdf5_path}: {e}")
            # Save the updated depth map as a PNG file
            #if datasource == "hypersim":
            try:
                updated_depth = hypersim_distance_to_depth(modified_depth)
                normalized_depth_map = cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_map_png = normalized_depth_map.astype(np.uint8)
            except Exception as e:
                print("Exception in trying to convert the depth from distance to depth using hypersim function")
                updated_depth = modified_depth
            normalized_depth_map = cv2.normalize(updated_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_png = normalized_depth_map.astype(np.uint8)
        else:
            depth_norm = cv2.normalize(modified_depth, None, 0, 65535, cv2.NORM_MINMAX)
            depth_map_png = depth_norm.astype(np.uint16)
        cv2.imwrite(output_depth_png_path, depth_map_png)

        print(f"  Processed {base_name}")

        with open(out_manifest_path, "a") as f:
            if depth_ext == ".hdf5":
                f.write(f"{out_img_path} {output_depth_hdf5_path} {output_depth_png_path}\n")
            else:
                f.write(f"{out_img_path} {output_depth_png_path}\n")


if __name__ == "__main__":
    main()
