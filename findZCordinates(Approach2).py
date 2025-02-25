import cv2
import numpy as np

# Load the images
left_image = cv2.imread('/Users/kushagravarshney/Desktop/ultraProject/project/sideView.jpeg', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('/Users/kushagravarshney/Desktop/ultraProject/project/reverseSideView.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded correctly
if left_image is None or right_image is None:
    print("Error: Could not load images.")
    exit()

# Ensure both images have the same size
if left_image.shape != right_image.shape:
    # Resize the images to match the smaller one
    height = min(left_image.shape[0], right_image.shape[0])
    width = min(left_image.shape[1], right_image.shape[1])
    left_image = cv2.resize(left_image, (width, height))
    right_image = cv2.resize(right_image, (width, height))

# Define the stereo matching parameters
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_image, right_image)

# Function to handle mouse click events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the disparity value at the clicked point
        disparity_value = disparity[y, x]

        # Assuming the camera parameters are known (focal length and baseline)
        focal_length = 0.8  # Example focal length in meters
        baseline = 0.5  # Example baseline distance in meters

        # Calculate the z-coordinate
        if disparity_value != 0:
            z = (((focal_length * baseline) / disparity_value) + 1000) 
            print(f"Clicked point: ({x}, {y})")
            print(f"Disparity value: {disparity_value}")
            print(f"Estimated z-coordinate: {z} meters")
        else:
            print("Disparity value is zero, cannot compute z-coordinate.")

# Display the left image and wait for a mouse click
cv2.imshow('Left Image', left_image)
cv2.setMouseCallback('Left Image', click_event)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load the images
# left_image = cv2.imread('/Users/kushagravarshney/Desktop/ultraProject/project/sideView.jpeg', cv2.IMREAD_GRAYSCALE)
# right_image = cv2.imread('/Users/kushagravarshney/Desktop/ultraProject/project/reverseSideView.jpeg', cv2.IMREAD_GRAYSCALE)

# # Check if images are loaded correctly
# if left_image is None or right_image is None:
#     print("Error: Could not load images.")
#     exit()

# # Ensure both images have the same size
# if left_image.shape != right_image.shape:
#     # Resize the images to match the smaller one
#     height = min(left_image.shape[0], right_image.shape[0])
#     width = min(left_image.shape[1], right_image.shape[1])
#     left_image = cv2.resize(left_image, (width, height))
#     right_image = cv2.resize(right_image, (width, height))

# # Define the stereo matching parameters
# numDisparities = 16 * 5  # Increase the number of disparities for better results
# blockSize = 15  # Block size for matching
# stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# # Compute the disparity map
# disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

# # Normalize the disparity map for better visualization
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # Apply a filter to remove noise
# filtered_disparity = cv2.filterSpeckles(disparity, 0, 4000, 16)

# # Convert filtered disparity back to 8-bit for display
# filtered_disparity_8bit = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # Function to handle mouse click events
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Get the disparity value at the clicked point
#         disparity_value = filtered_disparity[y, x]

#         # Assuming the camera parameters are known (focal length and baseline)
#         focal_length = 0.8  # Example focal length in meters
#         baseline = 0.5  # Example baseline distance in meters

#         # Calculate the z-coordinate
#         if disparity_value != 0 and disparity_value != 255:
#             z = (focal_length * baseline) / disparity_value
#             print(f"Clicked point: ({x}, {y})")
#             print(f"Disparity value: {disparity_value}")
#             print(f"Estimated z-coordinate: {z} meters")
#         else:
#             print("Disparity value is invalid, cannot compute z-coordinate.")

# # Display the left image and wait for a mouse click
# cv2.imshow('Left Image', left_image)
# cv2.imshow('Disparity Map', filtered_disparity_8bit)  # Display the filtered disparity map for debugging
# cv2.setMouseCallback('Left Image', click_event)

# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()