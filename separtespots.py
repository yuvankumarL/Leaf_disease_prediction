import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'TomatoEarlyBlight3.JPG'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use thresholding to create a binary image
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the leaf
contour = max(contours, key=cv2.contourArea)

# Create a mask for the leaf
mask = np.zeros_like(gray)
cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# Bitwise-and to cut out the leaf
leaf_only = cv2.bitwise_and(image, image, mask=mask)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.savefig("tomato/seperated_spots.png")


# Refine the approach by using color segmentation to better capture the leaf region.

# Convert to HSV color space for more precise color filtering
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for green (leaf color) in HSV space
lower_green = np.array([25, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask based on the color range
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Bitwise-and to cut out the leaf with improved mask
leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_cleaned)

# Display the refined result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(leaf_only_refined, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.savefig("tomato/check1.png")