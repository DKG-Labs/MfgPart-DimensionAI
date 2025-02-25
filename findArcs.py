# Import required libraries for image processing, file handling, and argument parsing
import cv2                    # OpenCV for image processing
import numpy as np           # Numerical operations
import os                    # Operating system interactions
from pathlib import Path     # Path handling
import csv                   # CSV file operations
import argparse              # Command-line argument parsing

# Set up base file paths using Path for cross-platform compatibility
base_path = Path(__file__).parent              # Get directory containing this script
image_path = base_path / "reverseTop.jpeg"     # Input image path
output_csv_path = base_path / "outputWithArcs.csv"  # Output CSV path
processed_image_path = base_path / "processedImageWithArcs.jpeg"  # Output annotated image path

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, 
                  min_arc_length=2, target_labels=None):
    """
    Process an image to detect and measure arc lengths, saving results to CSV and annotated image.
    
    Args:
        image_path: Path to input image
        reference_dimension: Known reference length for scaling
        output_csv: Path for output CSV file
        processed_image_path: Path for output annotated image
        min_arc_length: Minimum arc length to consider (default: 2)
        target_labels: Optional list of specific arc labels to include
    """
    # Validate image_path type
    if not isinstance(image_path, (str, Path)):
        raise TypeError(f"Expected string/Path for image_path, got {type(image_path)}")
    
    # Convert path to string and check if file exists
    image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found at {image_path}")
    
    # Load and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert to grayscale for edge detection
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image loaded and processed successfully.")
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # Detect edges using Canny algorithm
    edges = cv2.Canny(image_gray, 100, 200)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use largest contour as reference for scaling
    reference_contour = max(contours, key=cv2.contourArea)
    scaling_factor = reference_dimension / cv2.arcLength(reference_contour, True)
    
    # Initialize lists and variables for arc processing
    valid_arcs = []
    arc_index = 1
    annotated_image = image.copy()  # Create copy for annotation

    # Process each contour
    for contour in contours:
        # Calculate scaled arc length
        arc_length = cv2.arcLength(contour, False) * scaling_factor
        if arc_length >= min_arc_length:  # Filter by minimum length
            label = f"A{arc_index}"       # Generate label (A1, A2, etc.)
            valid_arcs.append((contour, label, arc_length))
            arc_index += 1

    # Filter arcs by target labels if specified
    if target_labels:
        valid_arcs = [arc for arc in valid_arcs if arc[1] in target_labels]

    # Prepare results list for CSV output
    results = []
    for contour, label, length in valid_arcs:
        results.append({"Arc Index": label, "Length": length})
        
        # Draw contour on annotated image in green
        cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
        
        # Calculate centroid for label placement
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            # Add label with calculated value
            text = f"{label}: {(length / 3.14) * 10:.2f}"
            cv2.putText(annotated_image, text, (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the annotated image
    cv2.imwrite(str(processed_image_path), annotated_image)
    print(f"Annotated image saved to {processed_image_path}")

    # Write results to CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["S.No.", "Index", "⌀ / 2()", "r.θ()"])
        writer.writeheader()
        for i, arc in enumerate(results, 1):
            writer.writerow({
                "S.No.": i,
                "Index": arc["Arc Index"],
                "⌀ / 2()": "",           # Empty field as not calculated
                "r.θ()": f"{arc['Length']:.2f}"
            })

    print(f"Found {len(results)} arcs meeting criteria")
    return results

# Main execution block
if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Measure arc lengths in an image')
    parser.add_argument('--labels', nargs='+', help='List of arc labels to include (e.g., A1 A2)')
    args = parser.parse_args()

    # Set reference dimension and call main function
    reference_dimension = 25.0
    get_dimensions(
        image_path=image_path,
        reference_dimension=reference_dimension,
        output_csv=output_csv_path,
        processed_image_path=processed_image_path,
        min_arc_length=2,
        target_labels=args.labels
    )