# import cv2
# import numpy as np

# # Load the image
# image_path = 'final_v2/test/Tomato_early_blight.JPG'
# image = cv2.imread(image_path)

# # Convert to HSV color space for more precise color filtering
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define color range for green (leaf color) in HSV space and adjust for better segmentation
# lower_green = np.array([20, 30, 30])  # Adjusted lower range
# upper_green = np.array([90, 255, 255])  # Adjusted upper range

# # Create a mask based on the color range for green areas
# mask_green = cv2.inRange(hsv, lower_green, upper_green)

# # Perform morphological operations to remove noise and fill gaps
# kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better cleaning
# mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
# mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Additional dilation to cover gaps

# # Find contours in the cleaned mask
# contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Define color ranges for black, brown, and orange (to detect dark spots)
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 50])  # Black spots (low brightness)
# lower_brown = np.array([10, 100, 20])
# upper_brown = np.array([20, 255, 200])  # Brown shades
# lower_orange = np.array([5, 50, 200])
# upper_orange = np.array([15, 255, 255])  # Orange shades

# # Create masks for each color range
# mask_black = cv2.inRange(hsv, lower_black, upper_black)
# mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
# mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

# # Combine masks for dark spots
# mask_dark_spots = cv2.bitwise_or(mask_black, mask_brown)
# mask_dark_spots = cv2.bitwise_or(mask_dark_spots, mask_orange)

# # Assume the largest contour is the leaf and create a mask from it
# if contours:
#     largest_contour = max(contours, key=cv2.contourArea)
#     mask_final = np.zeros_like(mask_cleaned)
#     cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)

#     # Combine the leaf mask and inverse of the dark spot mask
#     mask_exclude_dark = cv2.bitwise_and(mask_final, cv2.bitwise_not(mask_dark_spots))

#     # Create a pink background
#     background_pink = np.full(image.shape, (255, 182, 193), dtype=np.uint8)  # Pink background

#     # Add a pink outline to the leaf by creating a slightly larger contour
#     mask_outline = np.zeros_like(mask_final)
#     cv2.drawContours(mask_outline, [largest_contour], -1, 255, thickness=10)  # Adjust thickness as needed
#     outline_pink = cv2.bitwise_and(background_pink, background_pink, mask=mask_outline)
    
#     # Combine the pink background with the leaf-only image and the pink outline
#     leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_exclude_dark)
#     pink_background_image = cv2.bitwise_and(background_pink, background_pink, mask=~mask_final)  # Pink outside leaf
#     pink_background_image = cv2.add(pink_background_image, leaf_only_refined)  # Add the leaf region
#     pink_background_image = cv2.add(pink_background_image, outline_pink)  # Add the pink outline

# else:
#     pink_background_image = image  # If no leaf found, use original image

# # Save the result as a PNG image
# cv2.imwrite("mark/chk_pink/pink_background_excluding_dark_spots.png", pink_background_image)




import cv2
import numpy as np

# Load the image
image_path = 'final_v2/test/Tomato_early_blight.JPG'
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

    # Create a mask to exclude dark spots on the leaf
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_dark_spots = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # Detect dark areas

    # Combine the leaf mask and inverse of the dark spot mask
    mask_exclude_dark = cv2.bitwise_and(mask_final, mask_dark_spots)

    # Create a pink background
    background_pink = np.full(image.shape, (255, 182, 193), dtype=np.uint8)  # Pink background

    # Add a pink outline to the leaf by creating a slightly larger contour
    mask_outline = np.zeros_like(mask_final)
    cv2.drawContours(mask_outline, [largest_contour], -1, 255, thickness=10)  # Adjust thickness as needed
    outline_pink = cv2.bitwise_and(background_pink, background_pink, mask=mask_outline)
    
    # Combine the pink background with the leaf-only image and the pink outline
    leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_exclude_dark)
    pink_background_image = cv2.bitwise_and(background_pink, background_pink, mask=~mask_final)  # Pink outside leaf
    pink_background_image = cv2.add(pink_background_image, leaf_only_refined)  # Add the leaf region
    pink_background_image = cv2.add(pink_background_image, outline_pink)  # Add the pink outline

else:
    pink_background_image = image  # If no leaf found, use original image

# Save the result as a PNG image
cv2.imwrite("mark/chk_pink/pink_background_excluding_dark_spots.png", pink_background_image)
