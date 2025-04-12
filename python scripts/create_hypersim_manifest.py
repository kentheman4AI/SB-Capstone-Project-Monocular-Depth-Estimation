#Please give me a python script to do the following:
#1.  user selects a directory
#2.  go through the directory and look for all files that end with ".tonemap.jpg"
#3.  record the paths with the filenames in an array (tonemap), and sort the array in ascending order
#4.  go through the directory and look for all files that end with ".depth_meters.hdf5"
#5.  record the paths with the filenames in an array (depth_meters), and sort the array in ascending order
#6.  write out a ,txt file (hypersim_manifest.txt) with each line consisting of a single value from the tonemap array, followed by a space, and then a single value from the depth_meters array, in sequential order from both arrays


import os
from tkinter import Tk, filedialog

def find_and_sort_files(directory, extension):
    """
    Searches the directory for files with the specified extension and sorts them in ascending order.

    Args:
        directory (str): Directory to search.
        extension (str): File extension to search for.

    Returns:
        list: Sorted list of file paths with the specified extension.
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
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

    # Find and sort .tonemap.jpg files
    tonemap_files = find_and_sort_files(selected_directory, ".tonemap.jpg")

    # Find and sort .depth_meters.hdf5 files
    depth_meters_files = find_and_sort_files(selected_directory, ".depth_meters.hdf5")

    # Check if both arrays have the same length
    if len(tonemap_files) != len(depth_meters_files):
        print("Error: The number of tonemap and depth_meters files do not match.")
        return

    # Write out the hypersim_manifest.txt
    manifest_path = os.path.join(selected_directory, "hypersim_manifest.txt")
    with open(manifest_path, "w") as manifest_file:
        for tonemap, depth_meters in zip(tonemap_files, depth_meters_files):
            manifest_file.write(f"{tonemap} {depth_meters}\n")

    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()