import cv2
import numpy as np
import os
from pathlib import Path
import csv

# Relative paths
base_path = Path(__file__).parent
image_path = base_path / "image2.jpeg"
output_csv_path = base_path / "output_dimensions_filtered.csv"
processed_image_path = base_path / "processed_image.jpeg"

def calculate_length(contour):
    perimeter = cv2.arcLength(contour, True)
    return perimeter

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_radius=1):
    if not isinstance(image_path, (str, Path)):
        raise TypeError(f"Expected a string or Path for image_path, but got {type(image_path)}")
    
    image_path = str(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is in a supported format.")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image loaded and processed successfully.")
    
    # Apply Gaussian blur and edge detection
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(image_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Reference object for scaling factor
    reference_contour = max(contours, key=cv2.contourArea)
    scaling_factor = 0
    if reference_contour is not None and reference_dimension is not None:
        reference_object_length_pixels = calculate_length(reference_contour)
        if reference_object_length_pixels > 0:
            scaling_factor = reference_dimension / reference_object_length_pixels
        else: 
            raise ValueError("Reference object's contour has zero length.")
        
    # Step 1: Find the largest enclosing circle (outermost circle)
    largest_contour = max(contours, key=cv2.contourArea)
    (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(largest_contour)
    outer_radius_scaled = outer_radius * scaling_factor
    print(f"Outermost circle center: ({outer_x}, {outer_y}), Radius: {outer_radius_scaled:.2f} mm")
        
    # Store filtered results
    resultsForCircles = []

    # Make a copy of the image for annotation
    annotated_image = image.copy()

    circle_index = 1

    # Step 3: Process circles
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Scale radius
        unit = radius * scaling_factor

        # Calculate distance from center of outermost circle
        distance_from_outer_center = np.sqrt((x - outer_x)**2 + (y - outer_y)**2)
        
        # Filter out small circles using thresholds
        if unit >= min_radius and (distance_from_outer_center + radius) <= outer_radius:
            circle_label = f"C{circle_index}"
            if circle_label in ["C10"]:  # Only keep circles C7 and C10
                resultsForCircles.append({"Circle Index": circle_label, "Radius (mm)": unit})
                print(f"Circle {circle_index}: {unit:.2f} mm")

                # Draw the circle on the image
                center = (int(x), int(y))
                cv2.circle(annotated_image, center, int(radius), (255, 0, 0), 2)  # Blue for circles
                cv2.putText(annotated_image, circle_label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            circle_index += 1

    # Save the annotated image
    cv2.imwrite(processed_image_path, annotated_image)
    print(f"Processed image saved to {processed_image_path}")

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["S.No.", "Index", "\u2300 / 2(mm)", "r.\u03b8(mm)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write the data with serial number
        for index, result in enumerate(resultsForCircles, start=1):
            result_with_serial = {
                "S.No.": index,
                "Index": result.get("Circle Index"),
                "\u2300 / 2(mm)": result.get("Radius (mm)", ""),
                "r.\u03b8(mm)": ""
            }
            writer.writerow(result_with_serial)

    print(f"Filtered results saved to {output_csv}")
    print(f"Number of circles: {len(resultsForCircles)}")
    return resultsForCircles

# Example usage
reference_dimension = 25.0
get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, min_radius=1)
