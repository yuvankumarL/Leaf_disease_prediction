import cv2
import numpy as np

# Load the image and convert it to HSV color space
image = cv2.imread("pyfile/image_with_pink_background.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the specific BGR pink color (255, 182, 193) to HSV
pink_bgr = np.uint8([[[255, 182, 193]]])  # BGR color
pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)[0][0]

# Define a narrow range around the HSV value of pink
lower_pink = np.array([pink_hsv[0] - 10, 50, 50])   # Adjust as needed for tolerance
upper_pink = np.array([pink_hsv[0] + 10, 255, 255])

# Create a mask based on this specific pink color range
mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_pink, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Invert the mask to capture everything except the pink area
mask_background = cv2.bitwise_not(mask_cleaned)

# Save the mask_background as a PNG image with the same size as the original image
cv2.imwrite("mark/chk3/mask_background.png", mask_background)
