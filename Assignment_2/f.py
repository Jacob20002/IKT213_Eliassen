import cv2
import os

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Define the hsv function
def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the image to HSV
hsv_image = hsv(image)

# Save the image
cv2.imwrite('lena_hsv.png', hsv_image)
