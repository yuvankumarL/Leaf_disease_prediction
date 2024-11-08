import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "PotatoEarlyBlight4.JPG"  # Adjust this if needed
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary mask (you might need to adjust the threshold value)
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Invert to keep leaf area

# Create an all-white background
white_background = np.full_like(image_rgb, 255)  # White background (255 for R, G, and B)

# Use the mask to combine the leaf with the white background
leaf_on_white_bg = np.where(mask[:, :, None] == 255, image_rgb, white_background)

# Display the original and modified images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Leaf with white background
axes[1].imshow(leaf_on_white_bg)
axes[1].set_title("Leaf with White Background")
axes[1].axis("off")

# Show the images
plt.show()

# Save the result
plt.imsave("mark/leaf_white_background.png", leaf_on_white_bg)
