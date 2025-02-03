import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_arc_coordinates(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arc_contours = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) > 5:
            arc_contours.append(approx)
    
    arc_image = image.copy()
    for arc in arc_contours:
        cv2.drawContours(arc_image, [arc], -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(arc_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Arcs")
    plt.show()
    
    if arc_contours:
        largest_arc = max(arc_contours, key=cv2.contourArea)
        start_point = tuple(largest_arc[0][0])
        end_point = tuple(largest_arc[-1][0])
        return start_point, end_point
    else:
        return "No prominent arcs detected."

image_path = "/Users/kushagravarshney/Desktop/ultraProject/project/processedImageWithArcs.jpeg"
start, end = find_arc_coordinates(image_path)
print(f"Start Point: {start}, End Point: {end}")