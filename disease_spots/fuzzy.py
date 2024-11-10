import cv2
import numpy as np
from skimage import color
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Ensure output directory exists
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

# Define Fuzzy Logic System for Health Score
def define_fuzzy_system():
    # Define fuzzy variables
    disease_pct = ctrl.Antecedent(np.arange(0, 101, 1), 'disease_pct')
    health_score = ctrl.Consequent(np.arange(0, 101, 1), 'health_score')
    
    # Define membership functions for disease_pct
    disease_pct['low'] = fuzz.trimf(disease_pct.universe, [0, 0, 10])
    disease_pct['medium'] = fuzz.trimf(disease_pct.universe, [10, 20, 30])
    disease_pct['high'] = fuzz.trimf(disease_pct.universe, [30, 65, 100])
    
    # Define membership functions for health_score
    health_score['poor'] = fuzz.trimf(health_score.universe, [0, 35, 70])
    health_score['average'] = fuzz.trimf(health_score.universe, [70, 80, 90])
    health_score['good'] = fuzz.trimf(health_score.universe, [90, 100, 100])
    
    # Define fuzzy rules
    rule1 = ctrl.Rule(disease_pct['low'], health_score['good'])
    rule2 = ctrl.Rule(disease_pct['medium'], health_score['average'])
    rule3 = ctrl.Rule(disease_pct['high'], health_score['poor'])
    
    # Create control system and simulation
    health_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    health_sim = ctrl.ControlSystemSimulation(health_ctrl)
    
    return health_sim

# Calculate Health Score using Fuzzy Logic
def calculate_health_score(disease_percentage, health_sim):
    # Input the disease percentage
    health_sim.input['disease_pct'] = disease_percentage
    
    # Perform the computation
    health_sim.compute()
    
    # Get the output health score
    return health_sim.output['health_score']

# Main function to execute methods, calculate disease percentage, and health score
def process_image_methods(image_path):
    image = load_image(image_path)
    
    # Segment the leaf area to create a mask
    leaf_mask = segment_leaf(image)
    
    # Initialize fuzzy logic system
    health_sim = define_fuzzy_system()
    
    # Apply each method, mask the results, calculate disease percentage and health score
    for i, method in enumerate([method1, method2, method3, method4], start=1):
        binary_image = method(image, leaf_mask, os.path.join(output_dir, f'method{i}_result.jpg'))
            
        if i == 4:    
            disease_percentage = calculate_disease_percentage(binary_image, leaf_mask)
            health_score = calculate_health_score(disease_percentage, health_sim)
            
            print(f"Method {i} - Disease Percentage: {disease_percentage:.2f}% | Health Score: {health_score:.2f}/100")
    
    print("Disease spot detection images and health scores calculated successfully for all methods.")

# Run the function with the path to your input image
if __name__ == "__main__":
    process_image_methods('final_v2/test/new_folder_2/test/TomatoEarlyBlight2.JPG')
