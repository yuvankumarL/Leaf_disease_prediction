import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'PotatoEarlyBlight4.JPG'
image = cv2.imread(image_path)

# Convert to HSV color space for more precise color filtering
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for green (leaf color) in HSV space
lower_green = np.array([20, 30, 30])  # Adjusted lower range
upper_green = np.array([90, 255, 255])  # Adjusted upper range

# Create a mask based on the color range
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better cleaning
mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Additional dilation to cover gaps

# Find contours in the cleaned mask
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the leaf and create a mask from it
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    mask_final = np.zeros_like(mask_cleaned)
    cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Bitwise-and to cut out the leaf with the final mask
    leaf_only_refined = cv2.bitwise_and(image, image, mask=mask_final)
    
    # Create a pink background
    background_pink = np.full(image.shape, (255, 182, 193), dtype=np.uint8)  # Pink background

    # Combine the pink background with the leaf-only image
    pink_background_image = cv2.bitwise_and(background_pink, background_pink, mask=~mask_final)  # Pink outside leaf
    pink_background_image = cv2.add(pink_background_image, leaf_only_refined)  # Add the leaf region

    # Convert the leaf-only image to grayscale for spot detection
    leaf_gray = cv2.cvtColor(leaf_only_refined, cv2.COLOR_BGR2GRAY)

    # Define threshold to isolate black spots within the leaf
    _, black_region = cv2.threshold(leaf_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Mask the black regions to keep only spots within the leaf area
    black_in_leaf = cv2.bitwise_and(black_region, mask_final)

    # Highlight black spots on the leaf in red for easy identification
    highlighted_leaf = pink_background_image.copy()
    highlighted_leaf[black_in_leaf == 255] = [255, 0, 0]  # Set black spots to red
    
    # Display the result with pink background and red spots
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(highlighted_leaf, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Leaf with Pink Background and Black Spots Highlighted in Red")
    plt.show()
    plt.savefig("mark/seperate_spots_back.png")

else:
    print("No leaf detected.")
