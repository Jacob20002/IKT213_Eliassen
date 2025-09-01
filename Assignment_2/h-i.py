import cv2
import os

# Check current directory and file existence
print(os.getcwd())
print("lena-1.png" in os.listdir())

# Load the image
image = cv2.imread('lena-1.png')

# Define the smoothing function
def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# Blur the image with kernel size (15, 15)
smoothed_image = smoothing(image)

# Save the smoothed image
cv2.imwrite('lena_smoothed.png', smoothed_image)

# Define the rotation function
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image

# Rotate the image 90 degrees clockwise
rotated_image_90 = rotation(image, 90)

# Save the 90-degree rotated image
cv2.imwrite('lena_rotated_90.png', rotated_image_90)

# Rotate the image 180 degrees
rotated_image_180 = rotation(image, 180)

# Save the 180-degree rotated image
cv2.imwrite('lena_rotated_180.png', rotated_image_180)
