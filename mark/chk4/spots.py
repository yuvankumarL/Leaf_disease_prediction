import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert it to HSV color space
image1 = cv2.imread("mark/chk2/mask_background.png")
image2 = cv2.imread("mark/chk3/mask_background.png")

# Invert the mask to capture everything except the pink area
mask_background = cv2.bitwise_and(image1, image2)

# Save the mask_background as a PNG image
cv2.imwrite("mark/chk4/result.png", mask_background)

# # Display the result
# plt.figure(figsize=(10, 10))
# plt.imshow(mask_background, cmap="gray")
# plt.axis('off')
# plt.show()

# # Save the mask
# plt.savefig("mark/chk3/mask_background.png")
