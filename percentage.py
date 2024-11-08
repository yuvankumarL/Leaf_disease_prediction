# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load the image
# image_path = 'PotatoEarlyBlight4.JPG'
# image = cv2.imread(image_path)

# # Convert to HSV color space for more precise color filtering
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define color range for green (leaf color) in HSV space and adjust for better segmentation
# lower_green = np.array([20, 30, 30])  # Adjusted lower range
# upper_green = np.array([90, 255, 255])  # Adjusted upper range

# # Create a mask based on the color range
# mask_green = cv2.inRange(hsv, lower_green, upper_green)

# # Perform morphological operations to remove noise and fill gaps
# kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better cleaning
# mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
# mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Additional dilation to cover gaps

# # Find contours in the cleaned mask
# contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Assume the largest contour is the leaf and create a mask from it
# if contours:
#     largest_contour = max(contours, key=cv2.contourArea)
#     mask_final = np.zeros_like(mask_cleaned)
#     cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)

#     # Bitwise-and to cut out the leaf with the final mask
#     leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_final)
    
#     # Convert the leaf-only image to grayscale to focus on intensity
#     leaf_gray = cv2.cvtColor(leaf_only_refined, cv2.COLOR_BGR2GRAY)
    
#     # Define the threshold to consider a pixel as "black" or damaged (tunable threshold value)
#     _, black_region = cv2.threshold(leaf_gray, 50, 255, cv2.THRESH_BINARY_INV)  # Threshold to detect black regions

#     # Count the number of pixels in the leaf region (non-zero in mask_final)
#     total_leaf_pixels = cv2.countNonZero(mask_final)
    
#     # Count the number of black (damaged) pixels within the leaf region
#     black_leaf_pixels = cv2.countNonZero(cv2.bitwise_and(black_region, black_region, mask=mask_final))
    
#     # Calculate the percentage of black region within the leaf
#     black_region_percentage = (black_leaf_pixels / total_leaf_pixels) * 100 if total_leaf_pixels > 0 else 0

#     print(f"Percentage of black (damaged) region within the leaf: {black_region_percentage:.2f}%")
# else:
#     print("No leaf detected.")



import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'PotatoEarlyBlight4.JPG'
image = cv2.imread(image_path)

# Convert to HSV color space for more precise color filtering
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for green (leaf color) in HSV space and adjust for better segmentation
lower_green = np.array([20, 30, 30])  # Adjusted lower range
upper_green = np.array([90, 255, 255])  # Adjusted upper range

# Create a mask based on the color range
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better cleaning
mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Additional dilation to cover gaps

# Find contours in the cleaned mask
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the leaf and create a mask from it
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    mask_final = np.zeros_like(mask_cleaned)
    cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Bitwise-and to cut out the leaf with the final mask
    leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_final)
    
    # Convert the leaf-only image to grayscale to focus on intensity
    leaf_gray = cv2.cvtColor(leaf_only_refined, cv2.COLOR_BGR2GRAY)
    
    # Define the threshold to consider a pixel as "black" or damaged within the leaf
    _, black_region = cv2.threshold(leaf_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Mask the black regions to keep only the ones within the leaf area
    black_in_leaf = cv2.bitwise_and(black_region, mask_final)

    # Highlight black regions on the leaf image in red for visibility
    highlighted_leaf = leaf_only_refined.copy()
    highlighted_leaf[black_in_leaf == 255] = [255, 0, 0]  # Set black regions to red

    # Display the result with highlighted black regions
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(highlighted_leaf, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Leaf with Black Regions Highlighted in Red")
    plt.show()
    plt.savefig("tomato/spots_in_red.png")

    # Count the number of pixels in the leaf region (non-zero in mask_final)
    total_leaf_pixels = cv2.countNonZero(mask_final)
    
    # Count the number of black (damaged) pixels within the leaf region
    black_leaf_pixels = cv2.countNonZero(black_in_leaf)
    
    # Calculate the percentage of black region within the leaf
    black_region_percentage = (black_leaf_pixels / total_leaf_pixels) * 100 if total_leaf_pixels > 0 else 0

    print(f"Percentage of black (damaged) region within the leaf: {black_region_percentage:.2f}%")
else:
    print("No leaf detected.")
