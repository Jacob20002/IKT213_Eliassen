import cv2

def padding(image, border_width):
    """
    Add a reflected border around the image.

    Parameters:
    - image: Input image (as a NumPy array)
    - border_width: Width of the border to add on all sides

    Saves the padded image as 'padded_image.jpg'
    """
    padded_image = cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )

    # Save the result
    cv2.imwrite('padded_image.jpg', padded_image)
    print("Image saved as 'padded_image.jpg'.")

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("lena-1.png")  # Replace with your image file
    if image is None:
        print("Failed to load the image. Check the file path.")
    else:
        padding(image, border_width=100)
