import cv2
import numpy as np

# Load the images
image1 = cv2.imread("mark/chk3/mask_background.png")
image2 = cv2.imread("mark/chk5/pink_check1.png")

# Invert the first image
image1_not = cv2.bitwise_not(image1)
cv2.imwrite("mark/chk5/image1_not.png", image1_not)

# Apply bitwise OR to combine image2 with the inverted image1
mask_background = cv2.bitwise_or(image2, image1_not)
cv2.imwrite("mark/chk5/result.png", mask_background)

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
result = black_pixels / left * 100
print(f"Spot percentage: {result}")
