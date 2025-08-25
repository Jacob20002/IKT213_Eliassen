import cv2
import os

# Get the user's home directory and construct the full path
home_dir = os.path.expanduser("~")
output_dir = os.path.join(home_dir, "IKT213_Eliassen", "assignment_1", "solutions")
output_file = os.path.join(output_dir, "camera_outputs.txt")

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the default camera
cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
else:
    # Get video properties
    fps = cam.get(cv2.CAP_PROP_FPS)
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Save information to a text file
    with open(output_file, "w") as file:
        file.write("fps: {}\n".format(fps))
        file.write("height: {}\n".format(height))
        file.write("width: {}\n".format(width))

    print(f"Camera information saved to {output_file}")

    # Release the camera
    cam.release()
