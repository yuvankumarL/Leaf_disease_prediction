import cv2

# Load the image
image_path = "final/test/AppleCedarRust4.JPG"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Apply Bilateral Filter for noise reduction while keeping edges sharp
bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Convert the filtered image to grayscale for thresholding
gray_image = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to emphasize the leaf structure without too much blur
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Save the results
cv2.imwrite("noise/images/bilateral_filtered.png", bilateral_filtered)
cv2.imwrite("noise/images/thresholded_image.png", thresholded_image)

print("Images saved as bilateral_filtered.png and thresholded_image.png")
