import cv2
import numpy as np


def resize(image_path, scale_factor=2, up_or_down="up"):
    # Read the image
    image = cv2.imread(image_path)

    # Resize based on up_or_down parameter
    if up_or_down.lower() == "up":
        resized = cv2.pyrUp(image, dstsize=(int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    else:  # down
        resized = cv2.pyrDown(image, dstsize=(int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor)))

    # Save the output image
    output_path = 'resized_output.png'
    cv2.imwrite(output_path, resized)

    return resized


# Main execution
if __name__ == "__main__":
    input_image = "lambo.png"
    result = resize(input_image, 2, "up")