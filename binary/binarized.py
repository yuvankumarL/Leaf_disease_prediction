from PIL import Image
import numpy as np
from scipy.ndimage import median_filter

# Load the image
image_path = 'PotatoEarlyBlight4.JPG'  # Provide the path to your image
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')  # 'L' mode is for grayscale

# Convert the grayscale image to a NumPy array
image_array = np.array(gray_image)

# Define a threshold to binarize the image
threshold = 128  # You can adjust this threshold

# Apply threshold to create a binary image (0 or 255)
binary_image_array = (image_array > threshold) * 255  # Convert to 0 and 255

# Remove noise using median filter (adjust size as needed)
filtered_binary_image_array = median_filter(binary_image_array, size=3)  # size defines the kernel size

# Convert the filtered binary array back to an image
filtered_binary_image = Image.fromarray(filtered_binary_image_array.astype(np.uint8))

# Save the denoised binarized image
filtered_binary_image.save('binary/denoised_binarized_image.png')

# Optionally, display the denoised binarized image
filtered_binary_image.show()
