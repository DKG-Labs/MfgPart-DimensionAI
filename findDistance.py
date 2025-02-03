import cv2
import math

points = []
reference_length_mm = 1.5  
reference_pixels = None

def align_point(x, y, ref_x, ref_y):
    """Aligns the second point horizontally or vertically to the first point."""
    dx = abs(x - ref_x)
    dy = abs(y - ref_y)
    if dx > dy:
        return (x, ref_y)
    else:
        return (ref_x, y)

def click_event(event, x, y, flags, param):
    global points, reference_pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        if points:
            x, y = align_point(x, y, points[0][0], points[0][1])
        
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            pixel_distance = math.sqrt(dx**2 + dy**2)
            
            if reference_pixels is None:
                print(f"Reference distance: {pixel_distance:.2f} pixels = {reference_length_mm}")
                reference_pixels = pixel_distance
                points = []
            else:
                scale = reference_length_mm / reference_pixels
                mm_distance = pixel_distance * scale
                
                # Display the distance on the image
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                
                cv2.putText(img, f"{mm_distance * 10:.2f}", (mid_x, mid_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                print(f"Distance: {mm_distance * 10:.2f} ")
                
                points = []
                cv2.imshow("Image", img)

# Load the image
img = cv2.imread("/Users/kushagravarshney/Desktop/ultraProject/project/sideView.jpeg")
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("Instructions:")
print("1. Click TWO points on the REFERENCE OBJECT (e.g., a 10mm scale bar).")
print("2. Then, click TWO points to measure your target distance.")
print("3. The second point will automatically align to the closest horizontal or vertical axis.")
cv2.waitKey(0)
cv2.destroyAllWindows()