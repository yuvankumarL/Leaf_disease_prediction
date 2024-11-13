from rembg import remove
import easygui
from PIL import Image

# inputPath = easygui.fileopenbox(title='select image file')
# outputpath = easygui.fileopenbox(title='Save file to..')

input = Image.open("final_v2/test/AppleCedarRust4.JPG")
output = remove(input)
output.save("final_v4/remove_background.png")

