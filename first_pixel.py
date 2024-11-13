import cv2

# Load the image
image_path = "final_v5/highlighted_leaf_spots.png"  # Adjust this if needed
image = cv2.imread(image_path)

# Convert the image from BGR to RGB (OpenCV loads images in BGR format by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the color of the first pixel
first_pixel_color = image_rgb[0, 0]  # Access the top-left corner pixel

# Print the RGB values of the first pixel
print("First pixel color (RGB):", first_pixel_color)
