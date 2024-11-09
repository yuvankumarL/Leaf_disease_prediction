import cv2
import numpy as np



# Load the image and convert it to HSV color space
image1 = cv2.imread("mark/chk2/mask_background.png")
image2 = cv2.imread("mark/chk3/mask_background.png")

# Invert the mask to capture everything except the pink area
mask_background = cv2.bitwise_and(image1, image2)

# Convert to grayscale (if not already grayscale)
gray_mask = cv2.cvtColor(mask_background, cv2.COLOR_BGR2GRAY)

# Threshold the image to make sure we have only black and white
_, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

# Calculate the total number of pixels
total_pixels = binary_mask.size

# Count the number of white pixels
white_pixels = np.sum(binary_mask == 255)

# Count the number of black pixels in the binary mask (0 intensity)
black_pixels = np.sum(binary_mask == 0)

print(f"Total black pixels in image1: {black_pixels}")
print(f"Total white pixels in image1: {white_pixels}")
# Calculate the percentage of white pixels
percentage_white = (white_pixels / total_pixels) * 100

# Calculate the percentage of black pixels
percentage_black = (black_pixels / total_pixels) * 100

# Print the results
print(f"Percentage of white pixels: {percentage_white:.2f}%")
print(f"Percentage of black pixels: {percentage_black:.2f}%")

# Optionally save the mask
cv2.imwrite("mark/chk4/result.png", mask_background)

# Count black pixels in image1 and image2 (before any processing)
# Convert both images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Threshold the images to get black and white
_, binary_image1 = cv2.threshold(gray_image1, 127, 255, cv2.THRESH_BINARY)
_, binary_image2 = cv2.threshold(gray_image2, 127, 255, cv2.THRESH_BINARY)

# Count the black pixels in image1 and image2
black_pixels_image1 = np.sum(binary_image1 == 0)
black_pixels_image2 = np.sum(binary_image2 == 0)

white_pixels_image2 = np.sum(binary_image2 == 255)

# Print the total number of black pixels in image1 and image2
print(f"Total black pixels in image1: {black_pixels_image1}")
print(f"Total black pixels in image2: {black_pixels_image2}")
print(f"Total white pixels in image2: {white_pixels_image2}")