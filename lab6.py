# 
import cv2
import os
import numpy as np
import time

# Define the path to the image
real_path = '/media/phuc/D/4_years/8_sem/phuc/varotnikov/lab6/Lab6/image/'

# Read the template image
# template = cv2.imread(real_path + 'template.png', 0)
template = cv2.imread('test.png', 0)
print(os.path.isfile('test.png'))

# Check if the image was read successfully
if template is not None:
    # Display the image in a window
    cv2.imshow('import_template_oke', template)
    # Wait for 5 seconds
    time.sleep(5)
    # Close the window
    cv2.destroyWindow('import_template_oke')
else:
    # If there was an error reading the image, print an error message
    print('Error: Unable to read the image')
