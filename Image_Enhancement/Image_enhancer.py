# Import necessary libraries
import cv2
import numpy as np

# Load an image
image = cv2.imread("ONPIZA.jpg")

# Display the original image
cv2.imshow("Original Image", image)

# 1. Image Enhancement (Convert to Grayscale and Apply Gaussian Blur)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

cv2.imshow("Grayscale and Blurred Image", blurred_image)

# 2. Edge Detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)
cv2.imshow("Edge Detection", edges)

# 3. Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
keypoints_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Feature Detection (ORB)", keypoints_image)

# 4. Image Segmentation (Simple Thresholding)
_, threshold_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Segmentation", threshold_image)

# 5. Image Transformation (Resizing and Rotation)
# Resize the image
resized_image = cv2.resize(image, (200, 200))  # Resize to 200x200 pixels
cv2.imshow("Resized Image", resized_image)

# Rotate the image
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
cv2.imshow("Rotated Image", rotated_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
