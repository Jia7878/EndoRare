import os
import cv2
import numpy as np
import json

def get_grid_position(centroid, image_width, image_height):
    """
    Determines the grid position (1 to 9) based on the centroid coordinates.
    The grid numbering is as follows:

    1 | 2 | 3
    ---------
    4 | 5 | 6
    ---------
    7 | 8 | 9

    Returns:
        A string representing the grid position (e.g., "top-left", "center", etc.).
    """
    x, y = centroid
    # Define the boundaries for the grid
    third_width = image_width / 3
    third_height = image_height / 3

    if y < third_height:
        vertical = 'top'
    elif y < 2 * third_height:
        vertical = 'middle'
    else:
        vertical = 'bottom'

    if x < third_width:
        horizontal = 'left'
    elif x < 2 * third_width:
        horizontal = 'center'
    else:
        horizontal = 'right'

    if horizontal == 'left' and vertical == 'top':
        return 'top-left'
    elif horizontal == 'center' and vertical == 'top':
        return 'top-center'
    elif horizontal == 'right' and vertical == 'top':
        return 'top-right'
    elif horizontal == 'left' and vertical == 'middle':
        return 'middle-left'
    elif horizontal == 'center' and vertical == 'middle':
        return 'center'
    elif horizontal == 'right' and vertical == 'middle':
        return 'middle-right'
    elif horizontal == 'left' and vertical == 'bottom':
        return 'bottom-left'
    elif horizontal == 'center' and vertical == 'bottom':
        return 'bottom-center'
    elif horizontal == 'right' and vertical == 'bottom':
        return 'bottom-right'
    else:
        return 'unknown'

def generate_description(grid_position):
    """
    Generates an English description based on the grid position.
    """
    return f"located at the {grid_position} of the image."

def process_masks(mask_folder_path, output_json_path):
    """
    Processes mask images to extract their locations and writes the descriptions to a JSON file.

    Args:
        mask_folder_path (str): Path to the folder containing mask images.
        output_json_path (str): Path where the output JSON file will be saved.
    """
    data = {}

    # Iterate through all files in the mask folder
    for filename in os.listdir(mask_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(mask_folder_path, filename)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Unable to read image {filename}. Skipping.")
                continue

            # Threshold the mask to binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print(f"Warning: No contours found in image {filename}. Skipping.")
                continue

            # Assuming the largest contour corresponds to the mask
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate centroid
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                print(f"Warning: Zero division error for image {filename}. Skipping.")
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            image_height, image_width = binary_mask.shape

            # Determine grid position
            grid_position = get_grid_position((cX, cY), image_width, image_height)

            # Generate description
            description = generate_description(grid_position)

            # Populate the data dictionary
            data[filename] = {
                "Beard_and_Age": description
            }

            print(f"Processed {filename}: {description}")

    # Write data to JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"\nAll data has been written to {output_json_path}")

if __name__ == "__main__":
    # Define the path to the folder containing mask images
    mask_folder = "/home/hyl/yujia/A_few_shot/location/text_mask"  # Replace with your mask folder path

    # Define the output JSON file path
    output_json = "/home/hyl/yujia/A_few_shot/location/location.json"  # Replace with your desired output path

    process_masks(mask_folder, output_json)
