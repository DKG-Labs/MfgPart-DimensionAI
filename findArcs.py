import cv2
import numpy as np
import os
from pathlib import Path
import csv
import argparse

base_path = Path(__file__).parent
image_path = base_path / "reverseTop.jpeg"
output_csv_path = base_path / "outputWithArcs.csv"
processed_image_path = base_path / "processedImageWithArcs.jpeg"

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_arc_length=2, target_labels=None):
    if not isinstance(image_path, (str, Path)):
        raise TypeError(f"Expected string/Path for image_path, got {type(image_path)}")
    
    image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image loaded and processed successfully.")
    
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(image_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    reference_contour = max(contours, key=cv2.contourArea)
    scaling_factor = reference_dimension / cv2.arcLength(reference_contour, True)
    
    valid_arcs = []
    arc_index = 1
    annotated_image = image.copy()

    for contour in contours:
        arc_length = cv2.arcLength(contour, False) * scaling_factor
        if arc_length >= min_arc_length:
            label = f"A{arc_index}"
            valid_arcs.append((contour, label, arc_length))
            arc_index += 1

    if target_labels:
        valid_arcs = [arc for arc in valid_arcs if arc[1] in target_labels]

    results = []
    for contour, label, length in valid_arcs:
        results.append({"Arc Index": label, "Length": length})
        
        cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            text = f"{label}: {(length / 3.14) * 10:.2f}"
            cv2.putText(annotated_image, text, (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(processed_image_path), annotated_image)
    print(f"Annotated image saved to {processed_image_path}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["S.No.", "Index", "⌀ / 2()", "r.θ()"])
        writer.writeheader()
        for i, arc in enumerate(results, 1):
            writer.writerow({
                "S.No.": i,
                "Index": arc["Arc Index"],
                "⌀ / 2()": "",
                "r.θ()": f"{arc['Length']:.2f}"
            })

    print(f"Found {len(results)} arcs meeting criteria")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure arc lengths in an image')
    parser.add_argument('--labels', nargs='+', help='List of arc labels to include (e.g., A1 A2)')
    args = parser.parse_args()

    reference_dimension = 25.0
    get_dimensions(
        image_path=image_path,
        reference_dimension=reference_dimension,
        output_csv=output_csv_path,
        processed_image_path=processed_image_path,
        min_arc_length=2,
        target_labels=args.labels
    )