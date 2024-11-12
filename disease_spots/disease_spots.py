import cv2
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
import os

# Ensure output directory exists
output_dir = 'disease_spots/images'
os.makedirs(output_dir, exist_ok=True)

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Method 1: Grayscale + Simple Thresholding
def method1(image, output_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_path, binary_image)

# Method 2: Grayscale + Median Filtering + Thresholding
def method2(image, output_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_path, binary_image)

# Method 3: CIELAB Color Space, A Channel, Median Filtering + Thresholding
def method3(image, output_path):
    lab_image = color.rgb2lab(image)
    lab_image = (lab_image * 255 / np.max(lab_image)).astype(np.uint8)  # Scale to uint8
    a_channel = lab_image[:, :, 1]  # A channel
    filtered_a = cv2.medianBlur(a_channel, 5)
    _, binary_image = cv2.threshold(filtered_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_path, binary_image)

# Method 4: YCbCr Color Space, Cr Channel, Median Filtering + Thresholding
def method4(image, output_path):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    cr_channel = ycbcr_image[:, :, 1]  # Cr channel
    filtered_cr = cv2.medianBlur(cr_channel, 5)
    _, binary_image = cv2.threshold(filtered_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_path, binary_image)

# Main function to execute methods
def process_image_methods(image_path):
    image = load_image(image_path)
    
    # Applying each method and saving the result
    method1(image, os.path.join(output_dir, 'method1_result.jpg'))
    method2(image, os.path.join(output_dir, 'method2_result.jpg'))
    method3(image, os.path.join(output_dir, 'method3_result.jpg'))
    method4(image, os.path.join(output_dir, 'method4_result.jpg'))
    
    
    print("Disease spot detection images saved successfully for all methods.")

# Run the function with the path to your input image
process_image_methods('final_v2/test/Tomato_early_blight.JPG')
