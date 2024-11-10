import cv2
import numpy as np

from matplotlib import pyplot as plt

# Load the image
image_path = 'final_v2/test/PotatoEarlyBlight4.JPG'
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
    
    # Create a pink background
    background_pink = np.full(image.shape, (255, 182, 193), dtype=np.uint8)  # Pink background

    # Add a pink outline to the leaf by creating a slightly larger contour
    mask_outline = np.zeros_like(mask_final)
    cv2.drawContours(mask_outline, [largest_contour], -1, 255, thickness=10)  # Adjust thickness as needed
    outline_pink = cv2.bitwise_and(background_pink, background_pink, mask=mask_outline)
    
    # Combine the pink background with the leaf-only image and the pink outline
    pink_background_image = cv2.bitwise_and(background_pink, background_pink, mask=~mask_final)  # Pink outside leaf
    pink_background_image = cv2.add(pink_background_image, leaf_only_refined)  # Add the leaf region
    pink_background_image = cv2.add(pink_background_image, outline_pink)  # Add the pink outline

else:
    pink_background_image = image  # If no leaf found, use original image

cv2.imwrite("final_v2/images/new_folder/pink2.png", pink_background_image)


# Define the color range for white in BGR (use a threshold to capture shades of white)
lower_white = np.array([200, 200, 200])   # Lower bound for white
upper_white = np.array([255, 255, 255])   # Upper bound for white

# Create a mask to detect white areas
mask_white = cv2.inRange(pink_background_image, lower_white, upper_white)

# Create a pink color image of the same size as the input image
pink_color = np.full_like(pink_background_image, (255, 182, 193))  # Pink in BGR

# Use the mask to replace white areas with pink in the image
image_with_pink_background = cv2.bitwise_and(pink_color, pink_color, mask=mask_white)
image_with_pink_background = cv2.add(image_with_pink_background, cv2.bitwise_and(pink_background_image, pink_background_image, mask=~mask_white))

# Save the result as a PNG image
cv2.imwrite("final_v2/images/new_folder/pink.png", image_with_pink_background)


# Load the processed image (after applying all the masks and filters)
image = cv2.imread('final_v2/images/new_folder/pink.png')

# Convert the image to HSV for mask applications (if not already in HSV)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for identifying spots
# Define ranges based on your previous values
brown_range = ((10, 100, 20), (20, 255, 200))
black_range = ((0, 0, 0), (180, 255, 30))
orange_range = ((10, 100, 200), (20, 255, 255))
white_range = ((0, 0, 200), (180, 20, 255))
blue_range = ((100, 150, 0), (140, 255, 255))

# Create masks for spots (brown, black, orange, white)
brown_mask = cv2.inRange(hsv_image, brown_range[0], brown_range[1])
black_mask = cv2.inRange(hsv_image, black_range[0], black_range[1])
orange_mask = cv2.inRange(hsv_image, orange_range[0], orange_range[1])
white_mask = cv2.inRange(hsv_image, white_range[0], white_range[1])
blue_mask = cv2.inRange(hsv_image, blue_range[0], blue_range[1])

# Combine all spot masks
spot_mask = cv2.bitwise_or(brown_mask, black_mask)
spot_mask = cv2.bitwise_or(spot_mask, orange_mask)
spot_mask = cv2.bitwise_or(spot_mask, white_mask)
spot_mask = cv2.bitwise_or(spot_mask, blue_mask)

image = cv2.imread("final_v2/images/new_folder/pink.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the specific BGR pink color (255, 182, 193) to HSV
pink_bgr = np.uint8([[[255, 182, 193]]])  # BGR color
pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)[0][0]

# Define a narrow range around the HSV value of pink
lower_pink = np.array([pink_hsv[0] - 10, 50, 50])   # Adjust as needed for tolerance
upper_pink = np.array([pink_hsv[0] + 10, 255, 255])

# Create a mask based on this specific pink color range
mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_pink, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Invert the mask to capture everything except the pink area
leaf_mask = cv2.bitwise_not(mask_cleaned)

# Save the mask_background as a PNG image with the same size as the original image
cv2.imwrite("final_v2/images/new_folder/leaf_mask.png", leaf_mask)


# Calculate total leaf area (using the main leaf mask created previously)
# Assuming 'leaf_mask' is the binary mask for the entire leaf region
total_leaf_pixels = cv2.countNonZero(leaf_mask)

# Calculate spot area (using the combined spot mask)
spot_pixels = cv2.countNonZero(spot_mask)

# Calculate the ratio of spot area to leaf area
spot_ratio = spot_pixels / total_leaf_pixels if total_leaf_pixels != 0 else 0

# Print the results
print(f"Total Leaf Pixels: {total_leaf_pixels}")
print(f"Spot Pixels: {spot_pixels}")
print(f"Spot-to-Leaf Ratio: {spot_ratio:.2%}")
