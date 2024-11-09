import cv2
import numpy as np

# Load the image
image_path = "PotatoEarlyBlight4.JPG"  # Replace with your image path
image = cv2.imread(image_path)

# 1. Sharpen the image to reduce blur
# Define a sharpening kernel
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharp_image = cv2.filter2D(image, -1, sharpening_kernel)

# 2. Convert to HSV color space for color segmentation
hsv = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2HSV)

# 3. Adjust Contrast and Brightness (optional)
alpha = 1.2  # Contrast control
beta = 10    # Brightness control
adjusted_image = cv2.convertScaleAbs(sharp_image, alpha=alpha, beta=beta)

# 4. Smooth Noise (optional, if there is excess noise after sharpening)
# adjusted_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

# Now proceed with the original segmentation and highlighting steps:

# Define HSV ranges for yellowing and browning diseased leaves
# Yellowing areas (keep as yellow)
lower_yellowing = np.array([20, 30, 60])
upper_yellowing = np.array([35, 255, 255])

# Browning or dark spots (to be changed to red)
lower_browning = np.array([10, 30, 30])
upper_browning = np.array([20, 255, 200])

# Create masks for yellowing and browning
mask_yellowing = cv2.inRange(hsv, lower_yellowing, upper_yellowing)
mask_browning = cv2.inRange(hsv, lower_browning, upper_browning)

# Create a red color image for the browning areas
red_color = np.full_like(image, (0, 0, 255))  # Red in BGR

# Combine the image, keeping yellow areas as-is and turning dark spots red
highlighted_image = cv2.bitwise_and(adjusted_image, adjusted_image, mask=~mask_browning)  # Keep all except dark spots
highlighted_image = cv2.add(highlighted_image, cv2.bitwise_and(red_color, red_color, mask=mask_browning))  # Add red to dark spots

# Save the result
cv2.imwrite("highlighted_leaf.png", highlighted_image)

# Optional: Display the result
cv2.imshow("Highlighted Diseased Leaf", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
