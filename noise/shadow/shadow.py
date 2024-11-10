import cv2
import numpy as np

# Load the image
image = cv2.imread("final_v2/test/PotatoEarlyBlight4.JPG")

# Convert to grayscale to focus on illumination differences
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a large Gaussian blur to approximate the shadow areas
blurred = cv2.GaussianBlur(gray, (55, 55), 0)  # Adjust the kernel size if needed

# Avoid division by zero by adding a small value to the blurred image
blurred = blurred + 1

# Divide the original grayscale image by the blurred image to correct uneven illumination
corrected = (gray / blurred) * 255
corrected = np.clip(corrected, 0, 255).astype(np.uint8)  # Ensure values stay within 0-255

# Normalize the corrected image to further enhance contrast
corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

# Convert corrected grayscale back to BGR to match original color
shadow_removed_image = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

# Save the result
cv2.imwrite("noise/shadow/image_shadow_removed.png", shadow_removed_image)

# # Save the result
# cv2.imwrite("noise/shadow/image_shadow_removed.png", image_shadow_removed)
