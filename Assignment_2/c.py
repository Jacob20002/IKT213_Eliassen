import cv2
import os

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Define the resizing function
def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Resize the image to 200x200 pixels
resized_image = resize(image, 200, 200)

# Save the image
cv2.imwrite('lena_resized.png', resized_image)
