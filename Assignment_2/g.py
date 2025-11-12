import cv2
import os
import numpy as np

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Get height, width, and channels
height, width, channels = image.shape

# Create a new empty array
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

# Define the hue_shifted function
def hue_shifted(image, emptyPictureArray, hue):
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # Shift the color value by hue (50) and handle overflow/underflow
                new_value = image[y, x, c] + hue
                emptyPictureArray[y, x, c] = max(0, min(255, new_value))
    return emptyPictureArray

# Shift the image color value by 50
shifted_image = hue_shifted(image, emptyPictureArray, 50)

# Save the image
cv2.imwrite('lena_hue_shifted.png', shifted_image)
