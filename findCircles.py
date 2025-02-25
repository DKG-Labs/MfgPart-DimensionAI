import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import os  # Operating system interface
from pathlib import Path  # Object-oriented filesystem paths
import csv  # CSV file handling
import argparse  # Command-line argument parsing

# Define relative paths using Path for cross-platform compatibility
base_path = Path(__file__).parent  # Get the directory containing this script
image_path = base_path / "frontView.jpeg"  # Path to input image
output_csv_path = base_path / "outputWithCircles.csv"  # Path for CSV output
processed_image_path = base_path / "processedImageWithCircles.jpeg"  # Path for processed image output

def calculate_length(contour):
    """Calculate the perimeter length of a contour."""
    perimeter = cv2.arcLength(contour, True)  # True indicates closed contour
    return perimeter

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_radius=1, target_labels=None):
    """Detect circles in an image, calculate dimensions, and save results."""
    # Validate image_path type
    if not isinstance(image_path, (str, Path)):
        raise TypeError(f"Expected a string or Path for image_path, but got {type(image_path)}")
    
    image_path = str(image_path)  # Convert Path to string if necessary

    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found at {image_path}")
    
    # Load and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is in a supported format.")
    
    # Convert image to grayscale for processing
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image loaded and processed successfully.")

    # Apply Gaussian blur to reduce noise and improve edge detection
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # Detect edges using Canny edge detector
    edges = cv2.Canny(image_gray, 100, 200)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate scaling factor using the largest contour as reference
    reference_contour = max(contours, key=cv2.contourArea)  # Largest contour by area
    scaling_factor = 0
    if reference_contour is not None and reference_dimension is not None:
        reference_object_length_pixels = calculate_length(reference_contour)
        if reference_object_length_pixels > 0:
            scaling_factor = reference_dimension / reference_object_length_pixels  # Pixels to mm conversion
        else: 
            raise ValueError("Reference object's contour has zero length.")
        
    # Find the largest enclosing circle (outermost circle)
    largest_contour = max(contours, key=cv2.contourArea)
    (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(largest_contour)
    outer_radius_scaled = outer_radius * scaling_factor  # Scale radius to physical units (mm)
    print(f"Outermost circle center: ({outer_x}, {outer_y}), Radius: {outer_radius_scaled:.2f} mm")
        
    resultsForCircles = []  # List to store circle data
    annotated_image = image.copy()  # Copy of original image for annotations
    circle_index = 1  # Counter for labeling circles

    # Process each contour to find and annotate circles
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)  # Get minimum enclosing circle
        unit = radius * scaling_factor  # Convert radius to physical units (mm)
        distance_from_outer_center = np.sqrt((x - outer_x)**2 + (y - outer_y)**2)  # Distance from outer circle center
        
        # Filter circles: above minimum radius and within outermost circle
        if unit >= min_radius and (distance_from_outer_center + radius) <= outer_radius:
            circle_label = f"C{circle_index}"  # Label format: C1, C2, etc.
            if target_labels is None or circle_label in target_labels:  # Check if circle is in target list
                diameter_mm = 2 * unit  # Calculate diameter
                ds_value = f"(DS={unit * 10:.2f})"  # DS value (diameter scaled by 10)
                # Store circle data
                resultsForCircles.append({"Circle Index": circle_label, "Radius (mm)": unit, "Diameter (mm)": diameter_mm})
                print(f"Circle {circle_label}: Diameter {diameter_mm:.2f} mm")

                center = (int(x), int(y))  # Circle center as integers
                # Draw circle on annotated image in blue
                cv2.circle(annotated_image, center, int(radius), (255, 0, 0), 2)
                
                # Draw circle label in blue
                (text_width, _), _ = cv2.getTextSize(circle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(annotated_image, circle_label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw DS value in red next to label
                ds_position = (center[0] + text_width, center[1])
                cv2.putText(annotated_image, ds_value, ds_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            circle_index += 1  # Increment circle counter

    # Save the annotated image
    cv2.imwrite(processed_image_path, annotated_image)
    print(f"Processed image saved to {processed_image_path}")

    # Write results to CSV file
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["S.No.", "Index", "\u2300 / 2(mm)", "r.\u03b8(mm)"]  # CSV column headers (unicode for symbols)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write each circle's data to CSV
        for index, result in enumerate(resultsForCircles, start=1):
            result_with_serial = {
                "S.No.": index,  # Serial number
                "Index": result["Circle Index"],  # Circle label
                "\u2300 / 2(mm)": result["Radius (mm)"],  # Radius in mm
                "r.\u03b8(mm)": ""  # Placeholder for potential future use
            }
            writer.writerow(result_with_serial)

    print(f"Filtered results saved to {output_csv}")
    print(f"Number of circles: {len(resultsForCircles)}")
    return resultsForCircles  # Return list of circle data

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Detect circles in an image and output specified circles.')
    parser.add_argument('--labels', nargs='+', type=str, help='List of circle labels to include (e.g., C1 C2)')
    args = parser.parse_args()

    reference_dimension = 25.0  # Reference dimension in mm for scaling
    # Call the main function with specified parameters
    get_dimensions(
        image_path=image_path,
        reference_dimension=reference_dimension,
        output_csv=output_csv_path,
        processed_image_path=processed_image_path,
        min_radius=1,  # Minimum radius threshold
        target_labels=args.labels  # Optional list of target circle labels
    )