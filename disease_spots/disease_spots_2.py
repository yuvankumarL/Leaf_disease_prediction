import cv2
import numpy as np
from skimage import color
import os

output_dir = 'disease_spots/images'
os.makedirs(output_dir, exist_ok=True)

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Segment the leaf region to remove the white background
def segment_leaf(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Method 1: Grayscale + Simple Thresholding
def method1(image, mask, output_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    cv2.imwrite(output_path, binary_image)
    return binary_image

# Method 2: Grayscale + Median Filtering + Thresholding
def method2(image, mask, output_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    cv2.imwrite(output_path, binary_image)
    return binary_image

# Method 3: CIELAB Color Space, A Channel, Median Filtering + Thresholding
def method3(image, mask, output_path):
    lab_image = color.rgb2lab(image)
    lab_image = (lab_image * 255 / np.max(lab_image)).astype(np.uint8)
    a_channel = lab_image[:, :, 1]
    filtered_a = cv2.medianBlur(a_channel, 5)
    _, binary_image = cv2.threshold(filtered_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    cv2.imwrite(output_path, binary_image)
    return binary_image

# Method 4: YCbCr Color Space, Cr Channel, Median Filtering + Thresholding
def method4(image, mask, output_path):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    cr_channel = ycbcr_image[:, :, 1]
    filtered_cr = cv2.medianBlur(cr_channel, 5)
    _, binary_image = cv2.threshold(filtered_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    cv2.imwrite(output_path, binary_image)
    return binary_image

# Calculate the percentage of diseased spots based on leaf area
def calculate_disease_percentage(binary_image, leaf_mask):
    leaf_pixels = np.sum(leaf_mask > 0)
    diseased_pixels = np.sum(binary_image == 255)
    disease_percentage = (diseased_pixels / leaf_pixels) * 100 if leaf_pixels > 0 else 0
    return disease_percentage

# Main function to execute methods and calculate disease percentage
def process_image_methods(image_path):
    image = load_image(image_path)
    
    # Segment the leaf area to create a mask
    leaf_mask = segment_leaf(image)
    
    # Apply each method, mask the results, and calculate disease percentage
    for i, method in enumerate([method1, method2, method3, method4], start=1):
        binary_image = method(image, leaf_mask, os.path.join(output_dir, f'method{i}_result.jpg'))
        disease_percentage = calculate_disease_percentage(binary_image, leaf_mask)
        
        if i == 4:
            print(f"Method {i} - Disease percentage: {disease_percentage:.2f}%")
    
    print("Disease spot detection images saved successfully for all methods.")

# Run the function with the path to your input image
process_image_methods('final_v2/test/new_folder_2/Tomato___healthy.JPG')
