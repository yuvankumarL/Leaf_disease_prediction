from rembg import remove
from PIL import Image
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Load the image and remove the background
input_image = Image.open("final_v2/test/AppleCedarRust4.JPG")
foreground = remove(input_image)

# Step 2: Create a pink background and save the image with pink background
pink_background = Image.new("RGBA", foreground.size, (255, 182, 193, 255))
pink_background.paste(foreground, (0, 0), foreground)
output_path = "final_v5/pink_background.png"
pink_background.save(output_path)

# Step 3: Load the image with pink background and convert to HSV
image = cv2.imread(output_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 4: Define HSV ranges for colors to change to red
lower_brown = np.array([10, 30, 30])
upper_brown = np.array([25, 255, 200])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([20, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

# Step 5: Create masks for color ranges and combine them
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_black = cv2.inRange(hsv, lower_black, upper_black)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask_all_colors = cv2.bitwise_or(mask_brown, mask_black)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_orange)
mask_all_colors = cv2.bitwise_or(mask_all_colors, mask_white)

# Step 6: Highlight leaf spots in red
red_color = np.array([0, 0, 255], dtype=np.uint8)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
highlighted_image = np.copy(image_rgb)
highlighted_image[mask_all_colors > 0] = red_color
cv2.imwrite("final_v5/highlighted_leaf_spots.png", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))

# Step 7: Calculate the number of disease-affected (blue) pixels, leaf pixels, and pink background pixels
image = cv2.imread("final_v5/highlighted_leaf_spots.png")
height, width = image.shape[:2]
total_pixels = height * width
image_array = np.array(image)
num_blue_pixels = np.sum(np.all(image_array == (255, 0, 0), axis=-1))
num_pink_pixels = np.sum(np.all(image_array == (193, 182, 255), axis=-1))
num_leaf_pixels = total_pixels - num_pink_pixels

# Step 8: Calculate disease affected percentage
disease_affected = (int)((num_blue_pixels / num_leaf_pixels) * 100)
print(f"Disease affected percentage: {disease_affected}%")

# Step 9: Fuzzy logic for health score based on disease affected percentage
# Define fuzzy variables
disease_affected_input = ctrl.Antecedent(np.arange(0, 101, 1), 'disease_affected')
health_score_output = ctrl.Consequent(np.arange(0, 101, 1), 'health_score')

# Define fuzzy membership functions
disease_affected_input['low'] = fuzz.trimf(disease_affected_input.universe, [0, 0, 30])
disease_affected_input['medium'] = fuzz.trimf(disease_affected_input.universe, [20, 50, 60])
disease_affected_input['high'] = fuzz.trimf(disease_affected_input.universe, [50, 100, 100])

health_score_output['poor'] = fuzz.trimf(health_score_output.universe, [0, 0, 30])
health_score_output['average'] = fuzz.trimf(health_score_output.universe, [20, 50, 70])
health_score_output['good'] = fuzz.trimf(health_score_output.universe, [50, 100, 100])

# Define rules
rule1 = ctrl.Rule(disease_affected_input['low'], health_score_output['good'])
rule2 = ctrl.Rule(disease_affected_input['medium'], health_score_output['average'])
rule3 = ctrl.Rule(disease_affected_input['high'], health_score_output['poor'])

# Control system creation and simulation
health_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
health_simulation = ctrl.ControlSystemSimulation(health_ctrl)

# Set input and compute health score
health_simulation.input['disease_affected'] = disease_affected
health_simulation.compute()

# Output the health score
health_score = round(health_simulation.output['health_score'])
print(f"Health Score: {health_score}%")
