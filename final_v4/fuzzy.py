import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread("final_v2/test/new_folder_2/test/TomatoEarlyBlight2.JPG")
# Convert the image to
#  HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV ranges for colors to be changed to red
# Dark and Light Brown (HSV range for brown shades)
lower_brown = np.array([10, 30, 30])
upper_brown = np.array([25, 255, 200])

# Dark and Light Black (HSV range for black shades - low saturation and low value)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Orange (HSV range for orange shades)
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([20, 255, 255])

# White (HSV range for white shades - high value and low saturation)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

# Create masks for the specified color ranges
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_black = cv2.inRange(hsv, lower_black, upper_black)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Combine all masks (for brown, black, orange, white)
mask_all_colors = cv2.bitwise_or(mask_brown, mask_black)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_orange)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_white)

# Create a red color image for highlighting
red_color = np.array([0, 0, 255], dtype=np.uint8)  # Red color in BGR format

# Convert the image to RGB to match the expected output
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Change the selected colors to red
highlighted_image = np.copy(image_rgb)
highlighted_image[mask_all_colors > 0] = red_color

# Save and display the result
cv2.imwrite("final_v4/images/highlighted_leaf_spots.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV







# import cv2
# import numpy as np

# Load the image
image = cv2.imread("final_v4/images/highlighted_leaf_spots.png")

# Convert the image to HSV color space for easier color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV ranges for green and yellow-green to cover various leaf shades
lower_green = np.array([20, 30, 30])      # Range for green
upper_green = np.array([90, 255, 255])

lower_yellow_green = np.array([15, 50, 50])  # Range for yellowish-green
upper_yellow_green = np.array([35, 255, 255])

# Create masks for both green and yellowish-green regions
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_yellow_green = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

# Combine masks to capture both green and yellowish-green areas
mask_leaf = cv2.bitwise_or(mask_green, mask_yellow_green)

cv2.imwrite("final_v4/images/maks_leaf.png", mask_leaf)

# Create an image where the leaf region is black
highlighted_leaf_image = np.copy(image)
highlighted_leaf_image[mask_leaf > 0] = [0, 0, 0]  # Set leaf area to black

# Save the result
cv2.imwrite("final_v4/images/leaf_highlighted_black.png", highlighted_leaf_image)

xor = cv2.bitwise_or(highlighted_leaf_image, image)

cv2.imwrite("final_v4/images/xor.png", xor)




# Load the image
image = highlighted_leaf_image
height, width = image.shape[:2]
image_array = np.array(image)

# Calculate the total number of pixels
total_pixels = height * width

num_blue_pixels = np.sum(np.all(image_array == (255, 0, 0), axis=-1))
num_black_pixels = np.sum(np.all(image_array == (0, 0, 0), axis=-1))
# num_white_pixels = np.sum(np.all(image_array == (0, 0, 0), axis=-1))

print("Total Number of pixels:", total_pixels)
print("Number of blue pixels:", num_blue_pixels)
# print("Number of white pixels:", num_white_pixels)
print("Number of black pixels:", num_black_pixels)


score = (num_blue_pixels / num_black_pixels) * 100

print("disease affected score: ", score)





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



health_score = calculate_health_score(disease_percentage, health_sim)
            
print(f"Method {i} - Disease Percentage: {disease_percentage:.2f}% | Health Score: {health_score:.2f}/100")
