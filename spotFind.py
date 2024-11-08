import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Load the image
image_path = "PotatoEarlyBlight4.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_image)

# Apply Gaussian filter to the H channel to reduce noise
gaussian_blur = cv2.GaussianBlur(H, (9, 9), 0)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Filtered Image")
plt.show()

# Threshold the H channel to detect spots (adjust thresholds as necessary)
# These thresholds are approximate and may need adjustment for different images.
_, binary_spots = cv2.threshold(gaussian_blur, 30, 255, cv2.THRESH_BINARY_INV)

# Remove small noise by performing a morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_spots = cv2.morphologyEx(binary_spots, cv2.MORPH_OPEN, kernel)

# Count the number of spots
# Label connected components and count them
labels = measure.label(binary_spots, connectivity=2)
num_spots = np.max(labels)

# Display binary and labeled images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(binary_spots, cmap='gray')
plt.title("Binarized Spots Image")
plt.subplot(1, 2, 2)
plt.imshow(labels, cmap='nipy_spectral')
plt.title(f"Spots Counted: {num_spots}")
plt.show()
plt.savefig("plt.png")


print(f"Number of spots detected: {num_spots}")
