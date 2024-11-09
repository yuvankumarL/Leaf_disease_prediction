import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'pink.png'
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
else:
    leaf_only_refined = image  # In case no contours are found, keep the original image

# Create a pink background
background_pink = np.full(image.shape, (255, 182, 193), dtype=np.uint8)  # Pink background

# Apply pink background outside the leaf area
pink_background_image = cv2.bitwise_and(background_pink, background_pink, mask=~mask_final)  # Pink outside leaf
pink_background_image = cv2.add(pink_background_image, leaf_only_refined)

# Detect black spots within the leaf
lower_black = np.array([0, 0, 0])
upper_black = np.array([70, 70, 70])
black_spots_mask = cv2.inRange(image, lower_black, upper_black)
black_spots_within_leaf = cv2.bitwise_and(black_spots_mask, black_spots_mask, mask=mask_final)

# Overlay black spots on the pink background image
pink_background_image[black_spots_within_leaf > 0] = [0, 0, 0]  # Set black spot regions to black

# Display the final result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(pink_background_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Leaf with Pink Background and Black Spots Inside")
plt.show()
plt.savefig("mark/chk/separate_spots.png")
