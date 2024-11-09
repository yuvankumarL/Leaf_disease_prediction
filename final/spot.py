import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'final/test/AppleCedarRust4.JPG'
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

cv2.imwrite("final/images/pink2.png", pink_background_image)


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
cv2.imwrite("final/images/pink.png", image_with_pink_background)


hsv = cv2.cvtColor(image_with_pink_background, cv2.COLOR_BGR2HSV)

# Define color range for green (leaf color) in HSV space
lower_green = np.array([25, 40, 40])
upper_green = np.array([85, 255, 255])
# back_pink = np.array([255, 182, 193])

# Create a mask based on the color range
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Bitwise-and to cut out the leaf with improved mask
leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_cleaned)

# # Display the refined result
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(leaf_only_refined, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# plt.savefig("mark/chk5/pink_check1.png")

cv2.imwrite("final/images/pink_check1.png", leaf_only_refined)

hsv = cv2.cvtColor(image_with_pink_background, cv2.COLOR_BGR2HSV)

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

# # Invert the mask to capture everything except the pink area
# mask_background = cv2.bitwise_not(mask_cleaned)

# Save the mask_background as a PNG image with the same size as the original image
cv2.imwrite("final/images/mask_cleaned.png", mask_cleaned)



# Load the images
image1 = cv2.imread("final/images/mask_cleaned.png")
image2 = cv2.imread("final/images/pink_check1.png")

# # Invert the first image
# image1_not = cv2.bitwise_not(image1)
# cv2.imwrite("mark/chk5/image1_not.png", image1_not)

# Apply bitwise OR to combine image2 with the inverted image1
mask_background = cv2.bitwise_or(image2, image1)
cv2.imwrite("final/images/result.png", mask_background)

# Convert the result to grayscale
gray_mask = cv2.cvtColor(mask_background, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to obtain a binary mask
_, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

# Calculate the total number of pixels
total_pixels = mask_background.size

# Calculate the number of white and black pixels
white_pixels = np.sum(mask_background == 255)
black_pixels = np.sum(mask_background == 0)

# Print the results
print("Total number of pixels:", total_pixels)
print("Number of white pixels:", white_pixels)
print("Number of black pixels:", black_pixels)

left = (total_pixels-(white_pixels+black_pixels))
print(f"pixels of leaf without spots: {left}")
result = black_pixels / left * 100
print(f"Spot percentage: {result}")

