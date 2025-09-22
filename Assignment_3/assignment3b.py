import cv2
import numpy as np


def canny_edge_detection(image_path, threshold_1=50, threshold_2=50):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    # Save the output image
    output_path = 'canny_output.png'
    cv2.imwrite(output_path, edges)

    return edges


# Main execution
if __name__ == "__main__":
    input_image = "lambo.png"
    result = canny_edge_detection(input_image, 50, 50)
