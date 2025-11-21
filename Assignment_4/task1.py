import cv2
import numpy as np

# ---------------------------------
# 1. Harris Corner Detection
# ---------------------------------
def harris_corner_detection(reference_image_path):
    # Read the reference image
    img = cv2.imread(reference_image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"[ERROR] Could not read image at: {reference_image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate result for better visibility
    dst = cv2.dilate(dst, None)

    # Mark detected corners in red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Save the image with corners
    cv2.imwrite('harris.png', img)
    print("[OK] Harris corners saved as 'harris.png'")
    
    return img


# ---------------------------------
# Main execution
# ---------------------------------
if __name__ == "__main__":
    reference_path = "reference_img-1.png"
    harris_corner_detection(reference_path)
