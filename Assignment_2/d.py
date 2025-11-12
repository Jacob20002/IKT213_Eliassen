import cv2
import os
import numpy as np

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Get width, height, and channels
height, width, channels = image.shape

# Create a new empty array
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

# Define the copy function
def copy(image, emptyPictureArray):
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    return emptyPictureArray

# Copy the pixels from the image to the empty array
copied_image = copy(image, emptyPictureArray)

# Save the image
cv2.imwrite('lena_copied.png', copied_image)
