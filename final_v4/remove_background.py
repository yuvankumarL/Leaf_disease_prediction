# from rembg import remove
# import easygui
# from PIL import Image

# # inputPath = easygui.fileopenbox(title='select image file')
# # outputpath = easygui.fileopenbox(title='Save file to..')

# input = Image.open("final_v2/test/AppleCedarRust4.JPG")
# output = remove(input)
# output.save("final_v4/remove_background.png")

from rembg import remove
import easygui
from PIL import Image

# Open input image
input_image = Image.open("final_v2/test/Tomato_early_blight.JPG")

# Remove background
foreground = remove(input_image)

# Create a pink background image of the same size as the foreground
pink_background = Image.new("RGBA", foreground.size, (255, 182, 193, 255))  # RGBA for pink (255, 182, 193)

# Composite the pink background with the foreground
pink_background.paste(foreground, (0, 0), foreground)

# Save the output image with pink background
output_path = "final_v4/pink_background.png"
pink_background.save(output_path)
print(f"Image with pink background saved to {output_path}")


