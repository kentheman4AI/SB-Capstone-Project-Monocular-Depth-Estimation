import h5py
import numpy as np
import cv2
import os
from tkinter import Tk, filedialog

import argparse
import sys
import os
import h5py
import cv2
import numpy as np
from tkinter import Tk, filedialog


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

def hdf5_to_grayscale_image_with_dialog(dataset_name):
    """
    Allows the user to select an HDF5 file and converts a dataset to a grayscale image.

    Args:
        dataset_name (str): Name of the dataset within the HDF5 file to convert.
    """
    # Hide the root tkinter window
    Tk().withdraw()

    # Let the user select an HDF5 file
    hdf5_file_path = filedialog.askopenfilename(
        title="Select HDF5 File", 
        filetypes=[("HDF5 Files", "*.h5 *.hdf5")]
    )

    if not hdf5_file_path:
        print("No file selected. Exiting.")
        return

    # Extract the folder and base name
    folder = os.path.dirname(hdf5_file_path)
    base_name = os.path.splitext(os.path.basename(hdf5_file_path))[0]

    # Output image path
    output_image_path = os.path.join(folder, f"{base_name}_grayscale.png")

    # Open the HDF5 file
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        # Load the dataset
        if dataset_name not in hdf5_file:
            raise ValueError(f"Dataset '{dataset_name}' not found in the HDF5 file.")
        data = hdf5_file[dataset_name][()]

    depth = hypersim_distance_to_depth(data)
    # Normalize the data to range [0, 255]
    normalized_data = 255-cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX) #dark is 255:maximum depth, light is 0:minimum depth
    #normalized_data = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX) #dark is 0:minimum depth, light is 255:maximum depth

    # Convert to uint8 (required for image saving)
    grayscale_image = normalized_data.astype(np.uint8)

    # Save as a grayscale image
    cv2.imwrite(output_image_path, grayscale_image)
    print(f"Grayscale image saved to: {output_image_path}")

# Example Usage
dataset_name = "dataset"  # Change to the actual dataset name in your HDF5 file
hdf5_to_grayscale_image_with_dialog(dataset_name)
