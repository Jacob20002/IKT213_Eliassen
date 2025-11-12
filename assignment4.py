import cv2
import numpy as np

# ---------------------------------
# 1. Harris Corner Detection
# ---------------------------------
def harris_corner_detection(reference_image_path):
    # Read the reference image
    img = cv2.imread(reference_image_path)
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


# ---------------------------------
# 2. SIFT Feature-Based Alignment
# ---------------------------------
def align_images(image_to_align_path, reference_image_path, max_features=1000, good_match_percent=0.8):
    # Read both images
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    img = cv2.imread(image_to_align_path, cv2.IMREAD_COLOR)

    # Check for read errors
    if ref_img is None:
        print(f"[ERROR] Could not read reference image at: {reference_image_path}")
        return None
    if img is None:
        print(f"[ERROR] Could not read image to align at: {image_to_align_path}")
        return None

    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect features and compute descriptors
    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_gray, None)

    # Check that keypoints were found
    if descriptors1 is None or descriptors2 is None:
        print("[ERROR] No descriptors found. Try adjusting image or parameters.")
        return None

    # FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features using KNN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < good_match_percent * n.distance]
    print(f"[INFO] Found {len(good_matches)} good matches")

    # Draw matches (for visualization)
    im_matches = cv2.drawMatches(ref_img, keypoints1, img, keypoints2, good_matches, None)
    cv2.imwrite("matches.png", im_matches)
    print("[OK] Matches saved as 'matches.png'")

    # Need at least 4 good matches for homography
    if len(good_matches) < 4:
        print("[ERROR] Not enough matches to align images. Try increasing max_features or good_match_percent.")
        return None

    # Extract matched keypoints
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Compute homography using RANSAC
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp the second image to align with the first
    height, width, channels = ref_img.shape
    aligned = cv2.warpPerspective(img, h, (width, height))

    # Save the aligned image
    cv2.imwrite("aligned.png", aligned)
    print("[OK] Aligned image saved as 'aligned.png'")

    return aligned


# ---------------------------------
# 3. Run both parts
# ---------------------------------
if __name__ == "__main__":
    reference_path = "reference_img-1.png"
    image_to_align_path = "align_this.png"

    harris_corner_detection(reference_path)
    align_images(image_to_align_path, reference_path, max_features=1000, good_match_percent=0.8)