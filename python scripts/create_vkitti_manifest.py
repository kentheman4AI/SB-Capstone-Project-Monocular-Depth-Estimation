#Please give me a python script to do the following:
#1.  user selects a directory
#2.  go through the directory and look for all files that are named "rgb_xxxxx.jpg" with xxxxx being any five digits 
#3.  record the paths with the filenames in an array (rgb), and sort the array in ascending order
#4.  go through the directory and look for all files that end with "depth_xxxxx.png" with xxxxx being any five digits
#5.  record the paths with the filenames in an array (depth), and sort the array in ascending order
#6.  write out a ,txt file (vkitti_manifest.txt) with each line consisting of a single value from the rgb array, followed by a space, and then a single value from the depth array, in sequential order from both arrays


import os
import re
from tkinter import Tk, filedialog

def find_and_sort_files(directory, pattern):
    """
    Searches the directory for files matching the specified pattern and sorts them in ascending order.

    Args:
        directory (str): Directory to search.
        pattern (str): Regular expression pattern to match filenames.

    Returns:
        list: Sorted list of file paths matching the pattern.
    """
    file_list = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                file_list.append(os.path.join(root, file))
    return sorted(file_list)

def main():
    # Hide the root tkinter window
    Tk().withdraw()

    # Let the user select a directory
    selected_directory = filedialog.askdirectory(title="Select a Directory")

    if not selected_directory:
        print("No directory selected. Exiting.")
        return

    # Regular expressions for rgb and depth files
    rgb_pattern = r"rgb_\d{5}\.jpg"
    depth_pattern = r"depth_\d{5}\.png"

    # Find and sort rgb and depth files
    rgb_files = find_and_sort_files(selected_directory, rgb_pattern)
    depth_files = find_and_sort_files(selected_directory, depth_pattern)

    # Check if both arrays have the same length
    if len(rgb_files) != len(depth_files):
        print("Error: The number of rgb and depth files do not match.")
        return

    # Write out the vkitti_manifest.txt
    manifest_path = os.path.join(selected_directory, "vkitti_manifest.txt")
    with open(manifest_path, "w") as manifest_file:
        for rgb, depth in zip(rgb_files, depth_files):
            manifest_file.write(f"{rgb} {depth}\n")

    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
