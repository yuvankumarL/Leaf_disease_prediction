from rembg import remove
import easygui
from PIL import Image
import cv2
import numpy as np

# Open input image
input_image = Image.open("final_v2/test/AppleCedarRust4.JPG")

# Remove background
foreground = remove(input_image)

# Create a pink background image of the same size as the foreground
pink_background = Image.new("RGBA", foreground.size, (255, 182, 193, 255))  # RGBA for pink (255, 182, 193)

# Composite the pink background with the foreground
pink_background.paste(foreground, (0, 0), foreground)

# Save the output image with pink background
output_path = "final_v5/pink_background.png"
pink_background.save(output_path)
print(f"Image with pink background saved to {output_path}")






image = cv2.imread("final_v5/pink_background.png")
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

# Save and display the resu+lt
cv2.imwrite("final_v5/highlighted_leaf_spots.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV


image = cv2.imread("final_v5/highlighted_leaf_spots.png")
height, width = image.shape[:2]

total_pixels = height * width

image_array = np.array(image)

# Calculate the total number of pixels
total_pixels = height * width

num_blue_pixels = np.sum(np.all(image_array == (255, 0, 0), axis=-1))
num_pink_pixels = np.sum(np.all(image_array == (193, 182, 255), axis=-1))
num_leaf_pixels = total_pixels - num_pink_pixels

print("Total Number of pixels:", total_pixels)
print("Number of blue pixels:", num_blue_pixels)
print("Number of leaf pixels:", num_leaf_pixels)
print("Number of pink pixels:", num_pink_pixels)

disease_affected = (int)((num_blue_pixels / num_leaf_pixels) * 100)
print(f"Disease affected percentage:{disease_affected}%")