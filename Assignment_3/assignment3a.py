import cv2
import numpy as np


def sobel_edge_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)

    # Combine Sobel X and Y
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the result to 0-255
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = np.uint8(sobel_combined)

    # Save the output image
    output_path = 'sobel_output.png'
    cv2.imwrite(output_path, sobel_combined)

    return sobel_combined


# Main execution
if __name__ == "__main__":
    input_image = "lambo.png"
    result = sobel_edge_detection(input_image)
