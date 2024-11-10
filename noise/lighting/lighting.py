# import cv2
# import numpy as np

# def adjust_lighting(image_path, output_path, brightness_threshold=130):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Convert to grayscale to measure overall brightness
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mean_brightness = np.mean(gray)

#     # Check if the image is too bright
#     if mean_brightness > brightness_threshold:
#         # Convert to HSV to adjust brightness
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)
        
#         # Reduce brightness and increase contrast for bright images
#         v = cv2.equalizeHist(v)  # Equalize for better contrast
#         v = np.clip(v * 0.7, 0, 255).astype(np.uint8)  # Decrease brightness by 30%
        
#         # Merge and convert back to BGR
#         adjusted_hsv = cv2.merge((h, s, v))
#         adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
#     else:
#         # If brightness is acceptable, keep the image as is
#         adjusted_image = image

#     # Save the adjusted image
#     cv2.imwrite(output_path, adjusted_image)
#     print(f"Processed image saved at: {output_path}")


# import cv2
# import numpy as np

# def adjust_lighting_with_gamma(image_path, output_path, gamma=0.7):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Apply gamma correction
#     gamma_correction = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
#     adjusted_image = cv2.LUT(image, gamma_correction)

#     # Save the adjusted image
#     cv2.imwrite(output_path, adjusted_image)
#     print(f"Processed image saved at: {output_path}")

# # # Usage example
# # adjust_lighting_with_gamma('/mnt/data/AppleCedarRust1.jpg', '/mnt/data/adjusted_gamma_AppleCedarRust1.jpg')

# # Usage example
# adjust_lighting_with_gamma('final_v2/test/AppleCedarRust1.JPG', 'noise/lighting/adjusted_AppleCedarRust1.jpg')

import cv2

def adjust_lighting_with_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Load the image in grayscale
    image = cv2.imread(image_path)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into separate channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L-channel (lightness channel)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L-channel back with the A and B channels
    merged_lab = cv2.merge((cl, a_channel, b_channel))

    # Convert back to BGR color space
    adjusted_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Save the adjusted image
    cv2.imwrite(output_path, adjusted_image)
    print(f"Processed image saved at: {output_path}")

# Usage example
adjust_lighting_with_clahe('final_v2/test/AppleCedarRust1.JPG', 'noise/lighting/adjusted_AppleCedarRust1.jpg')


