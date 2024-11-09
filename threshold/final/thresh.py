# import cv2
# import numpy as np

# # Load the image
# image_path = "PotatoEarlyBlight4.JPG"  # Replace with your image path
# image = cv2.imread(image_path)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the HSV ranges for different colors that need to be turned red
# # Dark and Light Brown
# lower_brown = np.array([10, 30, 30])
# upper_brown = np.array([20, 255, 200])

# # Dark and Light Black (low saturation and value)
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 50])

# # Orange
# lower_orange = np.array([10, 50, 50])
# upper_orange = np.array([30, 255, 255])

# # White (low saturation, high value)
# lower_white = np.array([0, 0, 200])
# upper_white = np.array([180, 40, 255])

# # Create masks for the specified color ranges
# mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
# mask_black = cv2.inRange(hsv, lower_black, upper_black)
# mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
# mask_white = cv2.inRange(hsv, lower_white, upper_white)

# # Combine all masks (for brown, black, orange, white)
# mask_all_colors = cv2.bitwise_or(mask_brown, mask_black)
# mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_orange)
# mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_white)

# # Create a red color image for highlighting
# red_color = np.array([0, 0, 255], dtype=np.uint8)

# # Convert the image to RGB to match the expected output
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Change the selected colors to red
# highlighted_image = np.copy(image_rgb)
# highlighted_image[mask_all_colors > 0] = red_color

# # Save and display the result
# cv2.imwrite("threshold/final/highlighted_leaf.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV
# cv2.imshow("Highlighted Leaf", highlighted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the image
image_path = "final/images/pink.png"  # Replace with your image path
image = cv2.imread(image_path)

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
cv2.imwrite("threshold/final/tomato/highlighted_leaf_tomato.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV
cv2.imshow("Highlighted Leaf", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
