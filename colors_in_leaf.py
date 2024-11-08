import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image_path = "PotatoEarlyBlight4.JPG"  # Adjust this if needed
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Resize the image to speed up processing
image_resized = cv2.resize(image, (256, 256))

# Reshape the image to a 2D array of pixels
pixels = image_resized.reshape(-1, 3)

# Define the number of colors to identify
num_colors = 5

# Use KMeans clustering to find the dominant colors
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(pixels)

# Get the labels for each pixel (which color it belongs to)
labels = kmeans.labels_

# Get the colors (cluster centers)
colors = kmeans.cluster_centers_

# Create a segmented image where each pixel's color is replaced by its corresponding cluster center
segmented_image = colors[labels].reshape(image_resized.shape)

# Convert to uint8 type for visualization
segmented_image = np.uint8(segmented_image)

# Plot the original image and the segmented image side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original image
axes[0].imshow(image_resized)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Plot the segmented image with dominant colors
axes[1].imshow(segmented_image)
axes[1].set_title(f"Segmented Image ({num_colors} Colors)")
axes[1].axis("off")

# Show the plot
plt.show()

# Save the segmented image
plt.imsave("segmented_colors.png", segmented_image)
