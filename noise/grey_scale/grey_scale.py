import cv2

# Load the original image
image = cv2.imread('final_v2/test/Tomato_early_blight.JPG')  # Replace 'input_image.jpg' with your image file path

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save or display the grayscale image
cv2.imwrite('noise/grey_scale/gray_image.jpg', gray_image)  # This saves the grayscale image
# cv2.imshow('Grayscale Image', gray_image)  # This displays the grayscale image
# cv2.waitKey(0)  # Waits for a key press to close the displayed image
# cv2.destroyAllWindows()



# import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = gray_image

# Step 1: Thresholding to isolate the leaf
_, leaf_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 2: Find contours to get the leaf region
contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to separate the leaf region
leaf_only = np.zeros_like(image)
cv2.drawContours(leaf_only, contours, -1, 255, thickness=cv2.FILLED)

# Apply the leaf mask to the original image to isolate the leaf
leaf_region = cv2.bitwise_and(image, image, mask=leaf_only)

# Step 3: Detect disease spots (using adaptive thresholding or other techniques)
# Adaptive thresholding to highlight darker spots within the leaf region
disease_spots = cv2.adaptiveThreshold(leaf_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

# Filter out small spots to reduce noise
kernel = np.ones((3, 3), np.uint8)
disease_spots_cleaned = cv2.morphologyEx(disease_spots, cv2.MORPH_OPEN, kernel, iterations=2)

# # Display the results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original Grayscale Image")
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title("Leaf Region")
# plt.imshow(leaf_region, cmap='gray')
# plt.axis('off')

cv2.imwrite("noise/grey_scale/leaf_regions.png", leaf_region)

# plt.subplot(1, 3, 3)
# plt.title("Detected Disease Spots")
# plt.imshow(disease_spots_cleaned, cmap='gray')
# plt.axis('off')

cv2.imwrite("noise/grey_scale/disease_spots_cleaned.png", disease_spots_cleaned)
# plt.show()

