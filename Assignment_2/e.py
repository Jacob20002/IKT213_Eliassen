import cv2
import os

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Define the grayscale function
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to grayscale
gray_image = grayscale(image)

# Save the image
cv2.imwrite('lena_gray.png', gray_image)
