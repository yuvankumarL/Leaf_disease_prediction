import cv2
import numpy as np

# Load the image
image_path = "threshold/python/potato/pink.png"  # Replace with your image path
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV range for browning or dark spots (adjust based on your requirement)
lower_browning = np.array([10, 30, 30])
upper_browning = np.array([20, 255, 200])

# Create a mask for browning areas (dark spots)
mask_browning = cv2.inRange(hsv, lower_browning, upper_browning)

# Apply a binary threshold to focus on the diseased spots (browning areas)
_, binary_mask = cv2.threshold(mask_browning, 1, 255, cv2.THRESH_BINARY)

# Apply morphological operations to refine the mask (optional)
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

# Create the result image based on the binary mask
highlighted_image = np.copy(image)  # Copy the original image to modify it

# Iterate over each pixel in the image
for i in range(image.shape[0]):  # Loop over rows
    for j in range(image.shape[1]):  # Loop over columns
        if binary_mask[i, j] == 255:  # If the pixel is part of the dark spots
            highlighted_image[i, j] = [0, 0, 255]  # Change the pixel color to red

# Save and display the result
cv2.imwrite("threshold/python/potato/highlighted_leaf_red_spots_potato.png", highlighted_image)
cv2.imshow("Diseased Spots Highlighted in Red", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
