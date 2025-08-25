import cv2

def print_image_information(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size = image.size
    data_type = image.dtype

    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", size)
    print("data type:", data_type)

# Load the image
image = cv2.imread("lena-1.png")

# Check if image is loaded successfully
if image is not None:
    print_image_information(image)
else:
    print("Error: Could not load image 'lena-1.png'")
