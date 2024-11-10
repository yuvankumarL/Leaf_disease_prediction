import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'final_v2/test/TomatoHealthy2.JPG'
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
cv2.imwrite("final_v2/images/highlighted_blue_pixels.png", blue_mask) 







# Load the image
image_path = 'final_v2/images/highlighted_leaf_spots.png'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to HSV color space for more precise color filtering
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the blue color range in HSV space
lower_blue = np.array([100, 150, 50])   # Lower bound for blue
upper_blue = np.array([140, 255, 255])  # Upper bound for blue

# Create a mask based on the blue color range
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND the mask with the original image to extract only the blue regions
blue_regions = cv2.bitwise_and(image, image, mask=mask_blue)

# # Display the original image and the blue regions
# cv2.imshow("Original Image", image)
# cv2.imshow("Blue Regions", blue_regions)

# Save the result if needed
cv2.imwrite("final_v2/images/blue_regions.png", blue_regions)

# Convert image to RGB array
image_array = np.array(blue_regions)

# Define the target RGB value
target_rgb = (255, 0, 0)

# Count the number of pixels that match the target color
matching_pixels = np.sum(np.all(image_array == target_rgb, axis=-1))

print(f"matching_pixels: {matching_pixels}")

# Save it in a readable text format
output_path = 'final_v2/image_array_readable.txt'
np.savetxt(output_path, image_array.reshape(-1, image_array.shape[-1]), fmt='%d', delimiter=", ", header="R, G, B")




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


# blue = cv2.bitwise_or(pink_region, blue_regions)
# cv2.imwrite("final_v2/images/blue.png", blue)

# # Perform morphological operations to remove noise and fill gaps
# kernel = np.ones((5, 5), np.uint8)
# mask_cleaned = cv2.morphologyEx(mask_pink, cv2.MORPH_CLOSE, kernel)
# mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# # # Invert the mask to capture everything except the pink area
# # mask_background = cv2.bitwise_not(mask_cleaned)

# # Save the mask_background as a PNG image with the same size as the original image
# cv2.imwrite("final_v2/images/mask_cleaned.png", mask_cleaned)






# # Resize masks to match the shape of the blue_regions image
# mask_cleaned_resized = cv2.resize(mask_cleaned, (blue_regions.shape[1], blue_regions.shape[0]))
# # blue_mask_resized = cv2.resize(blue_mask, (blue_regions.shape[1], blue_regions.shape[0]))

# # Perform bitwise AND operation with resized masks
# blue_spots = cv2.bitwise_and(blue_regions, mask_cleaned_resized)
# cv2.imwrite("final_v2/images/blue_spots.png", blue_spots)


mask_cleaned_expanded = cv2.merge([mask_cleaned, mask_cleaned, mask_cleaned])
blue_mask_expanded = cv2.merge([blue_mask, blue_mask, blue_mask])

# Apply the bitwise operation with the expanded mask
blue_spots = cv2.bitwise_xor(blue_regions, mask_cleaned_expanded)

# If using just `mask_cleaned`, apply like this:
#blue_spots = cv2.bitwise_and(blue_regions, mask_cleaned_expanded)
cv2.imwrite("final_v2/images/blue_spots.png", blue_spots)

total_pixels = blue_spots.size
# white_pixels = np.sum(blue_spots == 255)
# black_pixels = np.sum(blue_spots == 0)

# # Print the results
# print("Total number of pixels:", total_pixels)
# print("Number of white pixels:", white_pixels)
# print("Number of black pixels:", black_pixels)

# left = (total_pixels-(white_pixels+black_pixels))
# print(f"pixels of leaf without spots: {left}")
# result = black_pixels / left * 100
# print(f"Spot percentage: {result}")


# Load the image
image = blue_spots
height, width = image.shape[:2]
image_array = np.array(image)

# Calculate the total number of pixels
total_pixels = height * width

# # Convert the image to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define color ranges in HSV
# # Blue color range
# lower_blue = np.array([100, 150, 0])
# upper_blue = np.array([140, 255, 255])

# # White color range
# lower_white = np.array([0, 0, 200])
# upper_white = np.array([180, 30, 255])

# # Black color range
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 50])

# # Create masks for each color
# blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# white_mask = cv2.inRange(hsv, lower_white, upper_white)
# black_mask = cv2.inRange(hsv, lower_black, upper_black)

# # Count the number of non-zero pixels in each mask
# num_blue_pixels = cv2.countNonZero(blue_mask)
# num_white_pixels = cv2.countNonZero(white_mask)
# num_black_pixels = cv2.countNonZero(black_mask)

num_blue_pixels = np.sum(np.all(image_array == (255, 0, 0), axis=-1))
num_black_pixels = np.sum(np.all(image_array == (255, 255, 255), axis=-1))
num_white_pixels = np.sum(np.all(image_array == (0, 0, 0), axis=-1))

print("Total Number of pixels:", total_pixels)
print("Number of blue pixels:", num_blue_pixels)
print("Number of white pixels:", num_white_pixels)
print("Number of black pixels:", num_black_pixels)


score = (num_blue_pixels / num_black_pixels) * 100

print("disease affected score: ", score)