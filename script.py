import cv2
import numpy as np
import os
from pathlib import Path
import csv

# Relative paths
base_path = Path(__file__).parent
image_path = base_path / "image8.jpg"
output_csv_path = base_path / "output_dimensions_filtered.csv"
processed_image_path = base_path / "processed_image.jpeg"

def calculate_length(contour):
    perimeter = cv2.arcLength(contour, True)
    return perimeter

# def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, selected_arcs=None, min_arc_length=2):
#     if selected_arcs is None:
#         selected_arcs = []
    
#     if not isinstance(image_path, (str, Path)):
#         raise TypeError(f"Expected a string or Path for image_path, but got {type(image_path)}")
    
#     image_path = str(image_path)

#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"File not found at {image_path}")
    
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is in a supported format.")
    
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     print("Image loaded and processed successfully.")
    
#     # Apply Gaussian blur and edge detection
#     image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
#     edges = cv2.Canny(image_gray, 100, 200)
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Reference object for scaling factor
#     reference_contour = max(contours, key=cv2.contourArea)
#     scaling_factor = 0
#     if reference_contour is not None and reference_dimension is not None:
#         reference_object_length_pixels = calculate_length(reference_contour)
#         if reference_object_length_pixels > 0:
#             scaling_factor = reference_dimension / reference_object_length_pixels
#         else: 
#             raise ValueError("Reference object's contour has zero length.")
        
#     # Store filtered results
#     resultsForArcs = []

#     # Make a copy of the image for annotation
#     annotated_image = image.copy()

#     arc_index = 1

#     # Step 2: Process arcs
#     for contour in contours:
#         arc_length = cv2.arcLength(contour, False)  # Arc length of the contour
#         arc = arc_length * scaling_factor
        
#         # Filter out small arcs using thresholds
#         if arc >= min_arc_length:
#             if f"A{arc_index}" in selected_arcs:  # Check if the current arc is selected
#                 resultsForArcs.append({"Arc Index": f"A{arc_index}", "Length of arc (mm)": arc})
#                 print(f"Arc {arc_index}: {arc:.2f} mm")

#                 # Draw the arc on the image
#                 cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)  # Green for arcs

#                 # Label the arc at a representative point
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
#                     cv2.putText(annotated_image, f"A{arc_index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             arc_index += 1

#     # Save the annotated image
#     cv2.imwrite(processed_image_path, annotated_image)
#     print(f"Processed image saved to {processed_image_path}")

#     with open(output_csv, mode='w', newline='') as csv_file:
#         fieldnames = ["S.No.", "Index", "⌀ / 2(mm)", "r.θ(mm)"]
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()

#         # Write the data with serial number
#         for index, result in enumerate(resultsForArcs, start=1):
#             result_with_serial = {
#                 "S.No.": index,
#                 "Index": result.get("Arc Index"),
#                 "⌀ / 2(mm)": result.get("Radius (mm)", ""),
#                 "r.θ(mm)": result.get("Length of arc (mm)", "")
#             }
#             writer.writerow(result_with_serial)

#     print(f"Filtered results saved to {output_csv}")
#     print(f"Number of arcs: {len(resultsForArcs)}")
#     return resultsForArcs

def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, min_radius=1, min_arc_length=2):
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
    resultsForArcs = []
    # resultsForCircles = []

    # Make a copy of the image for annotation
    annotated_image = image.copy()

    # circle_index = 1
    arc_index = 1

    # Step 2: Process arcs
    for contour in contours:
        arc_length = cv2.arcLength(contour, False)  # Arc length of the contour
        arc = arc_length * scaling_factor
        
        # Filter out small arcs using thresholds
        if arc >= min_arc_length:
            resultsForArcs.append({"Arc Index": f"A{arc_index}", "Length of arc (mm)": arc})
            print(f"Arc {arc_index}: {arc:.2f} mm")

            # Draw the arc on the image
            cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)  # Green for arcs

            # Label the arc at a representative point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(annotated_image, f"A{arc_index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            arc_index += 1

    # # Step 3: Process circles
    # for contour in contours:
    #     (x, y), radius = cv2.minEnclosingCircle(contour)

    #     # Scale radius
    #     unit = radius * scaling_factor

    #     # Calculate distance from center of outermost circle
    #     distance_from_outer_center = np.sqrt((x - outer_x)**2 + (y - outer_y)**2)
        
    #     # Filter out small circles using thresholds
    #     if unit >= min_radius and (distance_from_outer_center + radius) <= outer_radius:
    #         resultsForCircles.append({"Circle Index": f"C{circle_index}", "Radius (mm)": unit})
    #         print(f"Circle {circle_index}: {unit:.2f} mm")

    #         # Draw the circle on the image
    #         center = (int(x), int(y))
    #         cv2.circle(annotated_image, center, int(radius), (255, 0, 0), 2)  # Blue for circles
    #         cv2.putText(annotated_image, f"C{circle_index}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    #         circle_index += 1

    # Save the annotated image
    cv2.imwrite(processed_image_path, annotated_image)
    print(f"Processed image saved to {processed_image_path}")

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["S.No.", "Index", "⌀ / 2(mm)", "r.θ(mm)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # combined_results = resultsForCircles
        # combined_results = resultsForCircles + resultsForArcs
        combined_results = resultsForArcs
        
        # Write the data with serial number
        for index, result in enumerate(combined_results, start=1):
            result_with_serial = {
                "S.No.": index,
                "Index": result.get("Circle Index") or result.get("Arc Index"),
                "⌀ / 2(mm)": result.get("Radius (mm)", ""),
                "r.θ(mm)": result.get("Length of arc (mm)", "")
            }
            writer.writerow(result_with_serial)

    print(f"Filtered results saved to {output_csv}")
    print(f"Number of arcs: {len(resultsForArcs)}")
    # print(f"Number of circles: {len(resultsForCircles)}")
    # return resultsForCircles, resultsForArcs
    return resultsForArcs

# Example usage
reference_dimension = 25.0
# get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, min_radius=1, min_arc_length=2)
get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, min_arc_length=2)

# # Example usage
# reference_dimension = 25.0
# selected_arcs = ["A8"]  # Specify the arcs you want to label
# get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, selected_arcs=selected_arcs, min_arc_length=2)

# import cv2
# import numpy as np
# import os
# from pathlib import Path
# import csv

# # Relative paths
# base_path = Path(__file__).parent
# image_path = base_path / "image2.jpeg"
# output_csv_path = base_path / "output_dimensions_filtered.csv"
# processed_image_path = base_path / "processed_image.jpeg"

# def calculate_length(contour, start_point=None, end_point=None):
#     """Calculate the arc length of the contour or a segment of it."""
#     if start_point is not None and end_point is not None:
#         # Find the closest points on the contour to the given start and end points
#         start_idx = np.argmin(np.linalg.norm(contour[:, 0] - start_point, axis=1))
#         end_idx = np.argmin(np.linalg.norm(contour[:, 0] - end_point, axis=1))
        
#         # Extract the portion of the contour between start and end
#         if start_idx <= end_idx:
#             segment = contour[start_idx:end_idx + 1]
#         else:
#             # Handle wrap-around
#             segment = np.vstack((contour[start_idx:], contour[:end_idx + 1]))
        
#         return cv2.arcLength(segment, False)
#     else:
#         # Full contour length
#         return cv2.arcLength(contour, True)

# def get_dimensions(image_path, reference_dimension, output_csv, processed_image_path, selected_arcs=None, min_arc_length=2, start_point=None, end_point=None):
#     if selected_arcs is None:
#         selected_arcs = []  # Default to an empty list if no specific arcs are selected
    
#     if not isinstance(image_path, (str, Path)):
#         raise TypeError(f"Expected a string or Path for image_path, but got {type(image_path)}")
    
#     image_path = str(image_path)

#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"File not found at {image_path}")
    
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is in a supported format.")
    
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     print("Image loaded and processed successfully.")
    
#     # Apply Gaussian blur and edge detection
#     image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
#     edges = cv2.Canny(image_gray, 100, 200)
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Reference object for scaling factor
#     reference_contour = max(contours, key=cv2.contourArea)
#     scaling_factor = 0
#     if reference_contour is not None and reference_dimension is not None:
#         reference_object_length_pixels = calculate_length(reference_contour)
#         if reference_object_length_pixels > 0:
#             scaling_factor = reference_dimension / reference_object_length_pixels
#         else: 
#             raise ValueError("Reference object's contour has zero length.")
        
#     # Store filtered results
#     resultsForArcs = []

#     # Make a copy of the image for annotation
#     annotated_image = image.copy()

#     arc_index = 1

#     # Step 2: Process arcs
#     for contour in contours:
#         # Calculate arc length for full or partial contour
#         if start_point and end_point:
#             arc_length = calculate_length(contour, start_point, end_point)
#         else:
#             arc_length = calculate_length(contour)  # Full arc length

#         arc = arc_length * scaling_factor
        
#         # Filter out small arcs using thresholds
#         if arc >= min_arc_length:
#             if f"A{arc_index}" in selected_arcs:  # Check if the current arc is selected
#                 resultsForArcs.append({"Arc Index": f"A{arc_index}", "Length of arc (mm)": arc})
#                 print(f"Arc {arc_index}: {arc:.2f} mm")

#                 # Draw the arc on the image
#                 cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)  # Green for arcs

#                 # Label the arc at a representative point
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
#                     cv2.putText(annotated_image, f"A{arc_index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             arc_index += 1

#     # Save the annotated image
#     cv2.imwrite(processed_image_path, annotated_image)
#     print(f"Processed image saved to {processed_image_path}")

#     with open(output_csv, mode='w', newline='') as csv_file:
#         fieldnames = ["S.No.", "Index", "⌀ / 2(mm)", "r.θ(mm)"]
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()

#         # Write the data with serial number
#         for index, result in enumerate(resultsForArcs, start=1):
#             result_with_serial = {
#                 "S.No.": index,
#                 "Index": result.get("Arc Index"),
#                 "⌀ / 2(mm)": result.get("Radius (mm)", ""),
#                 "r.θ(mm)": result.get("Length of arc (mm)", "")
#             }
#             writer.writerow(result_with_serial)

#     print(f"Filtered results saved to {output_csv}")
#     print(f"Number of arcs: {len(resultsForArcs)}")
#     return resultsForArcs

# # Example usage
# reference_dimension = 25.0
# selected_arcs = ["A12"]  # Specify the arcs you want to label
# start_point = (800, 800)  # Replace with your start point coordinates
# end_point = (900, 900)    # Replace with your end point coordinates
# get_dimensions(image_path, reference_dimension, output_csv_path, processed_image_path, selected_arcs=selected_arcs, min_arc_length=2, start_point=start_point, end_point=end_point)