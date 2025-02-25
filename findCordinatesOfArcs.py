import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for displaying images

def find_arc_coordinates(image_path):
    """Detect arcs in an image and return the start and end coordinates of the largest arc."""
    # Load the image from the specified path
    image = cv2.imread(image_path)
    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection to identify edges in the grayscale image
    edges = cv2.Canny(gray, 50, 150)  # Thresholds 50 and 150 for edge detection
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_TREE retrieves all contours
    
    arc_contours = []  # List to store contours identified as arcs
    for cnt in contours:
        # Approximate the contour to a simpler polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)  # 2% of arc length as epsilon
        # Filter contours with more than 5 points as potential arcs
        if len(approx) > 5:  # More points suggest a smoother curve (arc-like)
            arc_contours.append(approx)
    
    # Create a copy of the original image for drawing detected arcs
    arc_image = image.copy()
    for arc in arc_contours:
        # Draw each detected arc contour in green (0, 255, 0) with thickness 2
        cv2.drawContours(arc_image, [arc], -1, (0, 255, 0), 2)
    
    # Set up a plot to display the image with detected arcs
    plt.figure(figsize=(6,6))  # Set figure size to 6x6 inches
    # Convert BGR (OpenCV format) to RGB (Matplotlib format) and display
    plt.imshow(cv2.cvtColor(arc_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Arcs")  # Add title to the plot
    plt.show()  # Display the plot
    
    # If any arcs were detected, process the largest one
    if arc_contours:
        # Find the contour with the largest area (assumed to be the most prominent arc)
        largest_arc = max(arc_contours, key=cv2.contourArea)
        # Extract start and end points from the approximated contour
        start_point = tuple(largest_arc[0][0])  # First point of the contour
        end_point = tuple(largest_arc[-1][0])  # Last point of the contour
        return start_point, end_point  # Return coordinates as tuples
    else:
        # Return a message if no arcs are detected
        return "No prominent arcs detected."

# Define the path to the input image
image_path = "/Users/kushagravarshney/Desktop/ultraProject/project/processedImageWithArcs.jpeg"
# Call the function and store the returned start and end points
start, end = find_arc_coordinates(image_path)
# Print the start and end coordinates of the largest detected arc
print(f"Start Point: {start}, End Point: {end}")