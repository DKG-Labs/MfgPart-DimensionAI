# Import necessary libraries
import cv2                # For image processing and computer vision tasks
import numpy as np        # For numerical operations and array handling
import os                 # For operating system interactions
from pathlib import Path  # For cross-platform path handling
import csv                # For CSV file operations

# Define relative paths using Path for platform independence
base_path = Path(__file__).parent                    # Get directory containing this script
image_path = base_path / "reverseSideView.jpeg"      # Input image file path
output_csv_path = base_path / "output_dimensions_filtered.csv"  # Output CSV file path
processed_image_path = base_path / "processed_image.jpeg"       # Output annotated image path

def calculate_length(contour):
    """Calculate the perimeter length of a contour."""
    perimeter = cv2.arcLength(contour, True)  # True indicates closed contour
    return perimeter

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_radius=1):
    """
    Process an image to detect circles, calculate dimensions, and annotate results.
    
    Args:
        image_path: Path to input image
        reference_dimension: Known reference length for scaling (in mm)
        output_csv: Path for CSV output file
        processed_image_path: Path for annotated image output
        min_radius: Minimum radius threshold for circle detection (default: 1)
    
    Returns:
        List of dictionaries containing circle data
    """
    # Validate image_path type
    if not isinstance(image_path, (str, Path)):
        raise TypeError(f"Expected a string or Path for image_path, but got {type(image_path)}")
    
    image_path = str(image_path)  # Convert Path to string if necessary

    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found at {image_path}")
    
    # Load image and verify successful loading
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is in a supported format.")
    
    # Convert to grayscale for processing
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image loaded and processed successfully.")
    
    # Apply preprocessing: blur to reduce noise, then edge detection
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)  # 5x5 kernel for smoothing
    edges = cv2.Canny(image_gray, 100, 200)              # Canny edge detection with thresholds
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find all contours
    
    # Calculate scaling factor using largest contour as reference
    reference_contour = max(contours, key=cv2.contourArea)
    scaling_factor = 0
    if reference_contour is not None and reference_dimension is not None:
        reference_object_length_pixels = calculate_length(reference_contour)
        if reference_object_length_pixels > 0:
            scaling_factor = reference_dimension / reference_object_length_pixels
        else: 
            raise ValueError("Reference object's contour has zero length.")
        
    # Find the largest enclosing circle (outermost boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(largest_contour)
    outer_radius_scaled = outer_radius * scaling_factor
    print(f"Outermost circle center: ({outer_x}, {outer_y}), Radius: {outer_radius_scaled:.2f} mm")
        
    # Initialize list to store circle results
    resultsForCircles = []
    annotated_image = image.copy()  # Create copy for drawing annotations
    circle_index = 1                # Counter for circle labeling

    # Process each contour to find and filter circles
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)  # Get minimum enclosing circle
        unit = radius * scaling_factor                    # Scale radius to physical units

        # Calculate distance from outermost circle's center
        distance_from_outer_center = np.sqrt((x - outer_x)**2 + (y - outer_y)**2)
        
        # Apply filters: minimum radius and must be inside outermost circle
        if unit >= min_radius and (distance_from_outer_center + radius) <= outer_radius:
            circle_label = f"C{circle_index}"
            if circle_label == "C10":  # Specifically filter for circle C10
                # Store circle data
                resultsForCircles.append({
                    "Circle Index": circle_label, 
                    "Radius (mm)": unit, 
                    "Center (x, y)": (x, y)
                })
                print(f"Circle {circle_label}: Center=({x:.2f}, {y:.2f}), Radius={unit:.2f} mm")

                # Annotate image: draw circle and add labels
                center = (int(x), int(y))
                cv2.circle(annotated_image, center, int(radius), (255, 0, 0), 2)  # Blue circle
                cv2.putText(annotated_image, circle_label, center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Blue label

                # Add center coordinates label
                center_label = f"Center ({int(x)}, {int(y)})"
                cv2.putText(annotated_image, center_label, (int(x) + 10, int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text
            circle_index += 1

    # Save annotated image to disk
    cv2.imwrite(processed_image_path, annotated_image)
    print(f"Processed image saved to {processed_image_path}")

    # Write results to CSV file
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["S.No.", "Index", "Radius (mm)", "Center (x, y)", "r.θ(mm)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write each result with a serial number
        for index, result in enumerate(resultsForCircles, start=1):
            result_with_serial = {
                "S.No.": index,
                "Index": result.get("Circle Index"),
                "Radius (mm)": result.get("Radius (mm)", ""),
                "Center (x, y)": result.get("Center (x, y)", ""),
                "r.θ(mm)": ""  # Empty field as not calculated
            }
            writer.writerow(result_with_serial)

    # Print summary
    print(f"Filtered results saved to {output_csv}")
    print(f"Number of circles: {len(resultsForCircles)}")
    return resultsForCircles

# Main execution block
if __name__ == "__main__":
    reference_dimension = 25.0  # Reference length in mm
    # Call function with predefined parameters
    get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, min_radius=1)