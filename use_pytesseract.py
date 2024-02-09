import pytesseract
from PIL import Image

# Open the image file
image = Image.open('raw/Foto am 05.02.24 um 14.31.jpg')

# Convert the image to grayscale
#image = image.convert('L')

# Recognize text using Tesseract
text = pytesseract.image_to_string(image, config='psm', lang='deu')

# Print the recognized text
print(text)

# Save the recognized text
with open('output/pytesseract_Foto am 05.02.24 um 14.31.jpg.txt', 'w') as f:
    f.write(text)
    f.close()