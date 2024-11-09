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

cv2.imwrite("final_v2/images/pink2.png", pink_background_image)


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
cv2.imwrite("final_v2/images/pink.png", image_with_pink_background)


# # Load the image
# image_path = "final/images/pink.png"  # Replace with your image path
# image = cv2.imread(image_path)

image = image_with_pink_background
# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV ranges for colors to be changed to red
# Dark and Light Brown (HSV range for brown shades)
lower_brown = np.array([10, 30, 30])
upper_brown = np.array([25, 255, 200])

# Dark and Light Black (HSV range for black shades - low saturation and low value)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Orange (HSV range for orange shades)
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([20, 255, 255])

# White (HSV range for white shades - high value and low saturation)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

# Create masks for the specified color ranges
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_black = cv2.inRange(hsv, lower_black, upper_black)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Combine all masks (for brown, black, orange, white)
mask_all_colors = cv2.bitwise_or(mask_brown, mask_black)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_orange)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_white)

# Create a red color image for highlighting
red_color = np.array([0, 0, 255], dtype=np.uint8)  # Red color in BGR format

# Convert the image to RGB to match the expected output
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Change the selected colors to red
highlighted_image = np.copy(image_rgb)
highlighted_image[mask_all_colors > 0] = red_color

# Save and display the result
cv2.imwrite("final_v2/images/highlighted_leaf_spots.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV
# cv2.imshow("Highlighted Leaf", highlighted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for blue color in HSV
lower_blue = np.array([85, 50, 50])   # Lower bound for blue color
upper_blue = np.array([130, 255, 255])  # Upper bound for blue color

# Create a mask for the blue pixels
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Create a red color image (BGR format)
red_color = np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR format

# Copy the original image to preserve other regions
highlighted_image = np.copy(image)

# Apply the red color to the masked blue regions
highlighted_image[blue_mask > 0] = red_color  # Replace blue pixels with red

# Save and display the result
cv2.imwrite("final_v2/images/highlighted_blue_pixels.png", highlighted_image) 