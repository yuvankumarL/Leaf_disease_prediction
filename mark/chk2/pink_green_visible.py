import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert it to HSV color space
image = cv2.imread("")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for green (leaf color) in HSV space
lower_green = np.array([25, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask based on the color range for green areas
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Invert the mask to capture the background (pink) area
mask_background = cv2.bitwise_not(mask_cleaned)

# Combine the green leaf area with the original background
leaf_area = cv2.bitwise_and(image, image, mask=mask_cleaned)
background_area = cv2.bitwise_and(image, image, mask=mask_background)
result = cv2.add(leaf_area, background_area)


# # Display the result
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.savefig("mark/chk2/leaf_area.png")

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(leaf_area, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.savefig("mark/chk2/result.png")


plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(background_area, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.savefig("mark/chk2/background_area.png")

# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(mask_background, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# plt.savefig("mark/chk2/mask_background.png")

cv2.imwrite("mark/chk2/mask_background.png", mask_background)