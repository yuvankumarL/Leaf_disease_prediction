import cv2
import numpy as np

# Load the image
image_path = "final/images/pink.png"  # Replace with your image path
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Threshold values for detecting dark and brown spots
# Dark spot detection (low value in V channel)
lower_dark_value = 50  # Threshold for dark spots in HSV
upper_dark_value = 100  # Upper limit for dark spots (black/brown)

# Brown color ranges (light and dark brown)
lower_brown = np.array([10, 30, 30])  # Lower range for brown
upper_brown = np.array([20, 255, 200])  # Upper range for brown

# Detect dark spots using Value channel (for dark colors like black and brown)
mask_dark_spots = cv2.inRange(hsv[:, :, 2], lower_dark_value, upper_dark_value)  # Based on V channel for darkness

# Detect brown spots (both light and dark) using H and S channels
mask_brown_spots = cv2.inRange(hsv, lower_brown, upper_brown)  # Using Hue and Saturation for brown

# Combine dark and brown spots masks
combined_mask = cv2.bitwise_or(mask_dark_spots, mask_brown_spots)

# Apply a binary threshold to focus on the diseased spots
_, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)

# Apply morphological operations to refine the mask (optional)
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

# Create a red color image
red_color = np.full_like(image, (0, 0, 255))  # Red in BGR

# Apply the red color only on the areas highlighted by the binary mask
highlighted_image = cv2.bitwise_and(red_color, red_color, mask=binary_mask)
highlighted_image = cv2.add(cv2.bitwise_and(image, image, mask=~binary_mask), highlighted_image)

# Save and display the result
cv2.imwrite("threshold/images/highlighted_leaf_red_spots_tom.png", highlighted_image)
cv2.imshow("Diseased Spots Highlighted in Red", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
