import cv2

# Load the image
image_path = "final/images/pink.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Apply Gaussian Blur to reduce Gaussian noise
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Median Blur to further reduce noise, especially salt-and-pepper noise
median_blur = cv2.medianBlur(gaussian_blur, 5)

# Convert the image to grayscale for thresholding
gray_image = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to emphasize the features of the leaf
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Save the results
cv2.imwrite("noise/gaussian_blur.png", gaussian_blur)
cv2.imwrite("noise/median_blur.png", median_blur)
cv2.imwrite("noise/thresholded_image.png", thresholded_image)

print("Images saved as gaussian_blur.png, median_blur.png, and thresholded_image.png")
