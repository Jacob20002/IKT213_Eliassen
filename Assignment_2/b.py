import cv2
import os

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Define the cropping function
def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

# Crop the image: 80 pixels from left (x_0=80), 130 pixels from right (x_1=width-130),
# 80 pixels from top (y_0=80), 130 pixels from bottom (y_1=height-130)
height, width = image.shape[:2]
cropped_image = crop(image, 80, width-130, 80, height-130)

# Save the image
cv2.imwrite('lena_cropped.png', cropped_image)
