import cv2
import numpy as np


def template_match(image_path, template_path):
    # Read the images
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    # Set threshold
    threshold = 0.9
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    # Draw rectangles around matches
    h, w = gray_template.shape
    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

    # Save the output image
    output_path = 'template_match_output.png'
    cv2.imwrite(output_path, image)

    return image


# Main execution
if __name__ == "__main__":
    input_image = "shapes.png"
    template_image = "shapes_template.jpg"
    result = template_match(input_image, template_image)