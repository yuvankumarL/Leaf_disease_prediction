import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image_path = "final_v2/test/new_folder_2/test/PotatoEarlyBlight1.JPG"  # Adjust this if needed
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Resize the image to speed up processing
image = cv2.resize(image, (256, 256))

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Define the number of colors to identify
num_colors = 10

# Use KMeans clustering to find the dominant colors
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(pixels)

# Get the colors and labels
colors = kmeans.cluster_centers_
labels = kmeans.labels_

# Function to print the lower and upper bounds for each cluster
def print_color_bounds(pixels, labels, colors, num_colors):
    for i in range(num_colors):
        # Get all pixels assigned to the current cluster
        cluster_pixels = pixels[labels == i]
        
        # Find the min and max values for each channel (R, G, B)
        lower_bound = cluster_pixels.min(axis=0)
        upper_bound = cluster_pixels.max(axis=0)
        
        print(f"Color {i + 1}:")
        print(f"  Lower Bound (RGB): {lower_bound}")
        print(f"  Upper Bound (RGB): {upper_bound}")
        print("-" * 50)

# Print the color bounds
print_color_bounds(pixels, labels, colors, num_colors)

# Plot the colors as a bar chart
def plot_colors(colors):
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0

    for i, color in enumerate(colors):
        end = start + 300 // num_colors
        rect[:, start:end] = color
        start = end
    
    plt.figure(figsize=(6, 2))
    plt.axis("off")
    plt.imshow(rect)
    plt.title("Dominant Colors")

# Plot the dominant colors
plot_colors(colors)

# Save the figure
plt.savefig("colors.png")  # Save the color plot image
