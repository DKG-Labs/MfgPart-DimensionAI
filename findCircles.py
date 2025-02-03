import cv2
import numpy as np
import os
from pathlib import Path
import csv
import argparse

# Relative paths
base_path = Path(__file__).parent
image_path = base_path / "frontView.jpeg"
output_csv_path = base_path / "outputWithCircles.csv"
processed_image_path = base_path / "processedImageWithCircles.jpeg"

def calculate_length(contour):
    perimeter = cv2.arcLength(contour, True)
    return perimeter

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_radius=1, target_labels=None):
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
        
    # Find the largest enclosing circle (outermost circle)
    largest_contour = max(contours, key=cv2.contourArea)
    (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(largest_contour)
    outer_radius_scaled = outer_radius * scaling_factor
    print(f"Outermost circle center: ({outer_x}, {outer_y}), Radius: {outer_radius_scaled:.2f} mm")
        
    resultsForCircles = []
    annotated_image = image.copy()
    circle_index = 1

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        unit = radius * scaling_factor
        distance_from_outer_center = np.sqrt((x - outer_x)**2 + (y - outer_y)**2)
        
        if unit >= min_radius and (distance_from_outer_center + radius) <= outer_radius:
            circle_label = f"C{circle_index}"
            if target_labels is None or circle_label in target_labels:
                diameter_mm = 2 * unit
                ds_value = f"(DS={unit * 10:.2f})"
                resultsForCircles.append({"Circle Index": circle_label, "Radius (mm)": unit, "Diameter (mm)": diameter_mm})
                print(f"Circle {circle_label}: Diameter {diameter_mm:.2f} mm")

                center = (int(x), int(y))
                cv2.circle(annotated_image, center, int(radius), (255, 0, 0), 2)
                
                # Draw circle label in blue
                (text_width, _), _ = cv2.getTextSize(circle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(annotated_image, circle_label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw DS value in green right after the label
                ds_position = (center[0] + text_width, center[1])
                cv2.putText(annotated_image, ds_value, ds_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            circle_index += 1

    cv2.imwrite(processed_image_path, annotated_image)
    print(f"Processed image saved to {processed_image_path}")

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["S.No.", "Index", "\u2300 / 2(mm)", "r.\u03b8(mm)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, result in enumerate(resultsForCircles, start=1):
            result_with_serial = {
                "S.No.": index,
                "Index": result["Circle Index"],
                "\u2300 / 2(mm)": result["Radius (mm)"],
                "r.\u03b8(mm)": ""
            }
            writer.writerow(result_with_serial)

    print(f"Filtered results saved to {output_csv}")
    print(f"Number of circles: {len(resultsForCircles)}")
    return resultsForCircles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect circles in an image and output specified circles.')
    parser.add_argument('--labels', nargs='+', type=str, help='List of circle labels to include (e.g., C1 C2)')
    args = parser.parse_args()

    reference_dimension = 25.0
    get_dimensions(
        image_path=image_path,
        reference_dimension=reference_dimension,
        output_csv=output_csv_path,
        processed_image_path=processed_image_path,
        min_radius=1,
        target_labels=args.labels
    )