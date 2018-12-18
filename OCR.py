# from PIL.Image import core as Image
from PIL import Image
# import Image
from pytesseract import image_to_string

print image_to_string(Image.open('test13.png'))
# print image_to_string(Image.open('test-english.jpg'), lang='eng')