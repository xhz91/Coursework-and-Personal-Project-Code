# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tPwnHqmPAydPd7uiy-cSQMOD_kg55wA4
"""

#pip install numpy opencv-python opencv-contrib-python
"""3a"""
import cv2
import numpy as np

# Read the image in color
img = cv2.imread('Frame1.png')

# create SIFT feature extractor
sift = cv2.SIFT_create()

# detect features from the image
keypoints, descriptors = sift.detectAndCompute(img, None)

# draw the detected key points
sift_image = cv2.drawKeypoints(img, keypoints, img)
# show the image
#cv2.imshow('image', sift_image)



# Read the image in color
img = cv2.imread('Frame2.png')

# create SIFT feature extractor
sift = cv2.SIFT_create()

# detect features from the image
keypoints, descriptors = sift.detectAndCompute(img, None)


# draw the detected key points
sift_image = cv2.drawKeypoints(img, keypoints, img)
# show the image
#cv2.imshow('image',sift_image)



"""3b"""

# Read the images
img1 = cv2.imread('Frame1.png')
img2 = cv2.imread('Frame2.png')

# Resize img2 to match img1 dimensions, if necessary
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Blend the two images with transparency to create an overlay
alpha = 0.5
overlay_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect SIFT features in both images
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# Create a feature matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors between both images
matches = bf.match(descriptors_1, descriptors_2)

# Sort the matches by distance and take the top 50 matches
matches = sorted(matches, key=lambda x: x.distance)

# Draw keypoints and lines on the overlay manually
for match in matches:
    # Get the matching keypoints for each image
    pt1 = tuple(np.round(keypoints_1[match.queryIdx].pt).astype(int))
    pt2 = tuple(np.round(keypoints_2[match.trainIdx].pt).astype(int))

    # Draw the keypoints in overlay image
    cv2.circle(overlay_img, pt1, 5, (0, 255, 0), -1)  # keypoints in img1
    cv2.circle(overlay_img, pt2, 5, (255, 0, 0), -1)  # keypoints in img2

    # Draw a line between the matched keypoints
    cv2.line(overlay_img, pt1, pt2, (255, 255, 0), 1)

# Display the centered overlay image with matches
#cv2.imshow('image',overlay_img)

"""3c"""
import numpy as np
# Step 1: Estimate the fundamental matrix from matched points
# Extract points from the matches
pts1 = np.array([keypoints_1[m.queryIdx].pt for m in matches], dtype=np.float32)
pts2 = np.array([keypoints_2[m.trainIdx].pt for m in matches], dtype=np.float32)

# Estimate the fundamental matrix from matched points using RANSAC
F_estimated, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
print("Fundamental Matrix from Matched Points:\n", F_estimated)

# Step 2: Calculate the fundamental matrix using camera parameters (example values below)
# Intrinsic matrix for Camera 1
K1 = np.array([[1.600e+03, 0.000e+00, 9.595e+02],
[0.000e+00, 1.600e+03, 5.395e+02],
[0.000e+00, 0.000e+00, 1.000e+00]])

# Intrinsic matrix for Camera 2
K2 = np.array([[1.49333333e+03, 0.00000000e+00, 9.78700000e+02],
[0.00000000e+00, 1.49333333e+03, 5.20300000e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Rotation and translation between Camera 1 and Camera 2
R = np.array([[ 0.9971792 , -0.00348069, 0.07497863],
[ 0.00362138, 0.99999203, -0.00174044],
[-0.07497205, 0.00200697, 0.99718366]])

T = np.array([-1.98989457, 0.00265269, 0.20979508])

# Compute the skew-symmetric matrix of T
#K2t = np.dot(K2,T)
#K2t_skew = np.array([
#    [0, -K2t[2], K2t[1]],
#    [K2t[2], 0, -K2t[0]],
#    [-K2t[1], K2t[0], 0]
#])
T_skew= np.array([
    [0, -T[2], T[1]],
    [T[2], 0, -T[0]],
    [-T[1], T[0], 0]
])

# Calculate the fundamental matrix using the camera parameters
#F_camera_try = K2t_skew @ K2 @ R @ np.linalg.inv(K1)
F_camera = np.linalg.inv(K2).T @ T_skew @ R @ np.linalg.inv(K1)

#print("\nFundamental Matrix from Camera Parameters:\n", F_camera_try)
print("\nFundamental Matrix from Camera Parameters:\n", F_camera)




"""3d"""
satisfying_matches = []

for i, match in enumerate(matches):
    # Extract the matched points
    x = np.array([*pts1[i], 1])  # Homogeneous coordinates for pts1
    x_prime = np.array([*pts2[i], 1])  # Homogeneous coordinates for pts2

    # Compute the epipolar constraint x'^T F x
    result = np.dot(x_prime.T, np.dot(F_camera, x))
    
    # Check if the constraint is satisfied (close to zero)
    if np.isclose(result, 0, atol=1e-3):  # Allow a small numerical tolerance
        satisfying_matches.append(match)

# Print the results
print(f"Total matches: {len(matches)}")
print(f"Satisfying matches: {len(satisfying_matches)}")


# Read the images
img1 = cv2.imread('Frame1.png')
img2 = cv2.imread('Frame2.png')

# Resize img2 to match img1 dimensions, if necessary
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Blend the two images with transparency to create an overlay
alpha = 0.5
overlay_img_satisfying = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

# Draw only the satisfying matches
for match in satisfying_matches:
    # Get the matching keypoints for each image
    pt1 = tuple(np.round(keypoints_1[match.queryIdx].pt).astype(int))
    pt2 = tuple(np.round(keypoints_2[match.trainIdx].pt).astype(int))

    # Draw the keypoints in overlay image
    cv2.circle(overlay_img_satisfying, pt1, 5, (0, 255, 0), -1)  # keypoints in img1 (green)
    cv2.circle(overlay_img_satisfying, pt2, 5, (255, 0, 0), -1)  # keypoints in img2 (red)

    # Draw a line between the matched keypoints
    cv2.line(overlay_img_satisfying, pt1, pt2, (255, 255, 0), 1)  # line color: cyan

# Display the overlay image with only satisfying matches
#cv2.imshow(overlay_img_satisfying)



"""3e"""
"""Estimate the area of swimming pool"""
""" 1. Compute Disarity map"""
# Read stereo images
img1 = cv2.imread('Frame1.png', 0)  # Load in grayscale
img2 = cv2.imread('Frame2.png', 0)

# Apply rectification in order to calculate disparity accurately
K1 = np.array([[1.600e+03, 0.000e+00, 9.595e+02],
[0.000e+00, 1.600e+03, 5.395e+02],
[0.000e+00, 0.000e+00, 1.000e+00]])

# Intrinsic matrix for Camera 2
K2 = np.array([[1.49333333e+03, 0.00000000e+00, 9.78700000e+02],
[0.00000000e+00, 1.49333333e+03, 5.20300000e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Distortion of Camera 1: 
D1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Distortion of Camera 2: 
D2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Rotation and translation between Camera 1 and Camera 2
R = np.array([[ 0.9971792 , -0.00348069, 0.07497863],
[ 0.00362138, 0.99999203, -0.00174044],
[-0.07497205, 0.00200697, 0.99718366]])

T = np.array([-1.98989457, 0.00265269, 0.20979508])

image_size = (1920, 1080)
"""
# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)

# Compute rectification maps
map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

# Load stereo images
img1 = cv2.imread('Frame1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Frame2.png', cv2.IMREAD_GRAYSCALE)

# Rectify images
rectified_img1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
rectified_img2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

# Calculate disparity
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(rectified_img1, rectified_img2)
"""

# Rectification transformation
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

# Rectification maps for both cameras
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

# Load the stereo images
img1 = cv2.imread('Frame1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Frame2.png', cv2.IMREAD_GRAYSCALE)

# Apply rectification
rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# Stereo block matching for disparity
stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=15)
disparity = stereo.compute(rectified_img1, rectified_img2)


depth = T[0]*K2[0][0]/disparity
#Point selected: (428, 656)
#Point selected: (630, 562)
#Point selected: (1071, 1002)
#Point selected: (1015, 641)
#Point selected: (1682, 258)

X1 = (428-K2[0][2])*K2[0][0]/disparity[656][428]+0.001
Y1 = (656-K2[1][2])*K2[0][0]/disparity[656][428]+0.001
X2 = (630-K2[0][2])*K2[0][0]/disparity[562][630]+0.001
Y2 = (562-K2[1][2])*K2[0][0]/disparity[562][630]+0.001
X3 = (1071-K2[0][2])*K2[0][0]/disparity[1002][1071]+0.001
Y3 = (1002-K2[1][2])*K2[0][0]/disparity[1002][1071]+0.001

X4 = (1015-K2[0][2])*K2[0][0]/disparity[641][1015]+0.001
Y4 = (641-K2[1][2])*K2[0][0]/disparity[641][1015]+0.001
X5 = (1682-K2[0][2])*K2[0][0]/disparity[258][1682]+0.001
Y5 = (258-K2[1][2])*K2[0][0]/disparity[258][1682]+0.001

width = np.sqrt((X2-X1)**2+(Y2-Y1)**2)
height = np.sqrt((X3-X2)**2+(Y3-Y2)**2)
area = width*height

print(area)

football_length = np.sqrt((X5-X4)**2+(Y5-Y4)**2)
print(football_length)


#import matplotlib as plt
## Display the rectified images
#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
#plt.title("Rectified Image 1")
#plt.imshow(rectified_img1, cmap='gray')
#plt.subplot(1, 2, 2)
#plt.title("Rectified Image 2")
#plt.imshow(rectified_img2, cmap='gray')
#plt.show()


"""Get the coordinates of the corners of the swimming pool"""
# Load the image
#img = cv2.imread('Frame1.png')
#
## List to store selected points
#selected_points = []
#
## Mouse callback function
#def select_point(event, x, y, flags, param):
#    global selected_points
#    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
#        selected_points.append((x, y))
#        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a small circle on the point
#        cv2.imshow('Select Points', img)
#        print(f"Point selected: ({x}, {y})")
#
## Display the image and set the mouse callback
#cv2.imshow('Select Points', img)
#cv2.setMouseCallback('Select Points', select_point)
#
## Wait until 'q' is pressed to finish selection
#print("Click on the corners of the swimming pool, then press 'q' to finish.")
#while True:
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cv2.destroyAllWindows()
#print("Selected Points:", selected_points)
#
##Point selected: (428, 656)
#Point selected: (630, 562)
#Point selected: (1071, 1002)
#Point selected: (1015, 641)
#Point selected: (1682, 258)
#"""
#"""Estimate the coordinates """
#
## Visible corner coordinates (replace these with actual values)
#A = np.array([429, 656])  # Top-left
#B = np.array([628, 563])  # Top-right
#C = np.array([1072, 1004])  # Bottom-left
#
## Calculate the missing corner D assuming a rectangular shape
#D = A + (C - B)
#
## Clamp coordinates to image bounds
#def clamp_coordinates(x, y, shape):
#    """Ensure coordinates are within image bounds."""
#    x = max(0, min(x, shape[1] - 1))  # width
#    y = max(0, min(y, shape[0] - 1))  # height
#    return int(x), int(y)
#
## Clamp the estimated D to valid bounds
#D = clamp_coordinates(D[0], D[1], disparity.shape)
#
## Convert pixel coordinates to 3D world coordinates
#def disparity_to_3d(x, y, disparity, focal_length, baseline, cx, cy):
#    """Convert pixel coordinates to 3D using disparity map and camera parameters."""
#    if not (0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]):
#        raise ValueError(f"Coordinates ({x}, {y}) are out of bounds.")
#    disp_value = disparity[int(y), int(x)]
#    if disp_value <= 0:
#        raise ValueError(f"Invalid disparity at ({x}, {y}): {disp_value}")
#    Z = (focal_length * baseline) / (disp_value + 1e-6)
#    X = (x - cx) * Z / focal_length
#    Y = (y - cy) * Z / focal_length
#    return np.array([X, Y, Z])
#
## Compute 3D Euclidean distance between two points
#def calculate_distance_3d(P1, P2):
#    """Compute the Euclidean distance between two 3D points."""
#    return np.linalg.norm(P2 - P1)
#
## Example parameters (replace with actual values)
#focal_length = 1600
#baseline = 2.0
#cx, cy = 959.5, 539.5
#
## Convert all corners to 3D
#try:
#    A_3d = disparity_to_3d(A[0], A[1], disparity, focal_length, baseline, cx, cy)
#    B_3d = disparity_to_3d(B[0], B[1], disparity, focal_length, baseline, cx, cy)
#    C_3d = disparity_to_3d(C[0], C[1], disparity, focal_length, baseline, cx, cy)
#    D_3d = disparity_to_3d(D[0], D[1], disparity, focal_length, baseline, cx, cy)
#except ValueError as e:
#    print(f"Error computing 3D coordinates: {e}")
#    exit()
#
## Compute the 3D side lengths
#try:
#    width = calculate_distance_3d(A_3d, B_3d)
#    height = calculate_distance_3d(A_3d, C_3d)
#except Exception as e:
#    print(f"Error computing distances: {e}")
#    exit()
#
## Estimate the area of the swimming pool
#area = width * height
#print(f"Estimated Swimming Pool Area: {area:.2f} square meters")