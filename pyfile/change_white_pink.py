import cv2
import numpy as np

# Load the image
image = cv2.imread('pyfile/pink3.png')

# Define the color range for white in BGR (use a threshold to capture shades of white)
lower_white = np.array([200, 200, 200])   # Lower bound for white
upper_white = np.array([255, 255, 255])   # Upper bound for white

# Create a mask to detect white areas
mask_white = cv2.inRange(image, lower_white, upper_white)

# Create a pink color image of the same size as the input image
pink_color = np.full_like(image, (255, 182, 193))  # Pink in BGR

# Use the mask to replace white areas with pink in the image
image_with_pink_background = cv2.bitwise_and(pink_color, pink_color, mask=mask_white)
image_with_pink_background = cv2.add(image_with_pink_background, cv2.bitwise_and(image, image, mask=~mask_white))

# Save the result
cv2.imwrite('pyfile/image_with_pink_background.png', image_with_pink_background)

# # Optional: Display the result
# cv2.imshow('Pink Background', image_with_pink_background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
