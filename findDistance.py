import cv2  # OpenCV library for image processing and interaction
import math  # Math library for distance calculations

# Global variables to store clicked points and reference pixel distance
points = []  # List to store coordinates of clicked points
reference_length_mm = 1.5  # Known length of the reference object in millimeters
reference_pixels = None  # Pixel distance of the reference object (to be calculated)

def align_point(x, y, ref_x, ref_y):
    """Aligns the second point horizontally or vertically to the first point based on the larger difference."""
    dx = abs(x - ref_x)  # Absolute difference in x-coordinates
    dy = abs(y - ref_y)  # Absolute difference in y-coordinates
    if dx > dy:  # If horizontal distance is greater, align y to reference
        return (x, ref_y)
    else:  # If vertical distance is greater, align x to reference
        return (ref_x, y)

def click_event(event, x, y, flags, param):
    """Handle mouse click events to mark points and calculate distances."""
    global points, reference_pixels  # Access global variables
    if event == cv2.EVENT_LBUTTONDOWN:  # Trigger on left mouse button click
        if points:  # If there's already a point, align the new one
            x, y = align_point(x, y, points[0][0], points[0][1])
        
        points.append((x, y))  # Add the clicked point to the list
        # Draw a red filled circle at the clicked point
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # -1 fills the circle
        
        if len(points) == 2:  # When two points are selected
            # Draw a blue line connecting the two points
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            
            # Calculate pixel distance between the two points
            dx = points[1][0] - points[0][0]  # Difference in x
            dy = points[1][1] - points[0][1]  # Difference in y
            pixel_distance = math.sqrt(dx**2 + dy**2)  # Euclidean distance in pixels
            
            if reference_pixels is None:  # First pair defines the reference
                # Print reference distance and set it
                print(f"Reference distance: {pixel_distance:.2f} pixels = {reference_length_mm}")
                reference_pixels = pixel_distance  # Store reference pixel distance
                points = []  # Reset points for the next measurement
            else:  # Subsequent pairs calculate actual distances
                scale = reference_length_mm / reference_pixels  # Scaling factor (mm/pixel)
                mm_distance = pixel_distance * scale  # Convert pixel distance to mm
                
                # Calculate midpoint for text placement
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                
                # Display the distance (scaled by 10) on the image in blue
                cv2.putText(img, f"{mm_distance * 10:.2f}", (mid_x, mid_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                print(f"Distance: {mm_distance * 10:.2f} ")  # Print distance
                
                points = []  # Reset points for the next measurement
                cv2.imshow("Image", img)  # Update the displayed image

# Load the image from the specified path
img = cv2.imread("/Users/kushagravarshney/Desktop/ultraProject/project/sideView.jpeg")
cv2.imshow("Image", img)  # Display the initial image
cv2.setMouseCallback("Image", click_event)  # Set mouse callback to handle clicks

# Print usage instructions
print("Instructions:")
print("1. Click TWO points on the REFERENCE OBJECT (e.g., a 10mm scale bar).")
print("2. Then, click TWO points to measure your target distance.")
print("3. The second point will automatically align to the closest horizontal or vertical axis.")

cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows