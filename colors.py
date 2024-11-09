import cv2
import numpy as np
import webcolors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image_path = "PotatoEarlyBlight4.JPG"  # Replace with your image path
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

# Get the colors (centroids)
colors = kmeans.cluster_centers_

# Function to find the closest color name
def closest_color(requested_color):
    min_colors = {}
    for name, rgb in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(rgb)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[rd + gd + bd] = name
    return min_colors[min(min_colors.keys())]

# Print the dominant color names based on the RGB values
for i, color in enumerate(colors):
    color_name = closest_color(color)  # Find the closest color name
    print(f"Color {i+1}: {color_name}, RGB: {color}")

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

# Plot and save the dominant colors
plot_colors(colors)
plt.savefig("dominant_colors.png")  # Save the bar chart of the dominant colors
