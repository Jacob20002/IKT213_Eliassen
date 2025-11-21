import cv2
import numpy as np

def align_images(image_to_align, reference_image, max_features=1500, good_match_percent=0.15, aligned_path="aligned.png", matches_path="matches.png"):
    # Load images
    reference_img = cv2.imread(reference_image)
    image_to_align_img = cv2.imread(image_to_align)
    
    if reference_img is None or image_to_align_img is None:
        print("Could not open or find the images.")
        return

    # Convert to grayscale
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    image_to_align_gray = cv2.cvtColor(image_to_align_img, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using ORB
    orb_detector = cv2.ORB_create(nfeatures=max_features)
    reference_keypoints, reference_descriptors = orb_detector.detectAndCompute(reference_gray, None)
    image_keypoints, image_descriptors = orb_detector.detectAndCompute(image_to_align_gray, None)
    
    print(f"[INFO] Detected {len(reference_keypoints)} keypoints in reference image")
    print(f"[INFO] Detected {len(image_keypoints)} keypoints in image to align")
    
    if reference_descriptors is None or image_descriptors is None:
        return
    if len(reference_keypoints) < 4 or len(image_keypoints) < 4:
        return

    # Match features using brute force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_matches = matcher.knnMatch(image_descriptors, reference_descriptors, k=2)

    # Apply Lowe's ratio test to filter good matches
    quality_matches = []
    ratio_threshold = 0.75
    for match_pair in all_matches:
        if len(match_pair) == 2:
            first_match, second_match = match_pair
            if first_match.distance < ratio_threshold * second_match.distance:
                quality_matches.append(first_match)
    
    if len(quality_matches) < 4:
        return

    print(f"[INFO] Found {len(quality_matches)} good matches after ratio test")

    # Sort matches by distance and select top matches
    quality_matches = sorted(quality_matches, key=lambda match: match.distance)
    
    num_matches_to_use = max(10, int(len(all_matches) * good_match_percent))
    selected_matches = quality_matches[:min(len(quality_matches), num_matches_to_use)]
    
    print(f"[INFO] Using {len(selected_matches)} matches for homography estimation")

    # Extract matched point coordinates
    source_points = np.float32([image_keypoints[match.queryIdx].pt for match in selected_matches])
    destination_points = np.float32([reference_keypoints[match.trainIdx].pt for match in selected_matches])
    
    # Compute homography matrix
    homography_matrix, inlier_mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    if homography_matrix is None:
        return
    
    inliers = np.sum(inlier_mask) if inlier_mask is not None else len(selected_matches)
    print(f"[INFO] Homography computed with {inliers} inliers")

    # Warp the image to align with reference
    ref_height, ref_width = reference_img.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align_img, homography_matrix, (ref_width, ref_height))

    # Save aligned image
    cv2.imwrite(aligned_path, aligned_image)
    print(f"[OK] Aligned image saved as '{aligned_path}'")
    
    # Create visualization of matches
    inlier_mask_list = inlier_mask.ravel().tolist() if inlier_mask is not None else None
    match_visualization = cv2.drawMatches(
        image_to_align_img, image_keypoints, 
        reference_img, reference_keypoints, 
        selected_matches, None,
        matchColor=(0, 255, 0),
        matchesMask=inlier_mask_list,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(matches_path, match_visualization)
    print(f"[OK] Matches visualization saved as '{matches_path}'")


if __name__ == "__main__":
    align_images(
        "align-this.jpg",
        "reference_img-1.png",
        max_features=1500,
        good_match_percent=0.15
    )
