import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for displaying images and handling clicks
import csv  # CSV file handling
import os  # Operating system interface for file paths

# Define the path to the input image
image_path = "/Users/kushagravarshney/Desktop/ultraProject/project/reverseSideView.jpeg"
image = cv2.imread(image_path)  # Load the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for edge detection

# Detect edges using the Canny edge detector
edges = cv2.Canny(gray, 50, 150)  # Thresholds 50 and 150 for edge detection

# Detect lines using Probabilistic Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
# Parameters: rho=1 pixel, theta=1 degree, threshold=100 votes, minLineLength=100 pixels, maxLineGap=10 pixels

# Get image dimensions and calculate center
height, width = gray.shape  # Height and width of the grayscale image
center_x, center_y = width // 2, height // 2  # Center coordinates of the image

horizontal_line = None  # Variable to store the detected horizontal line
threshold = 15  # Threshold for how close a line must be to the center to be considered horizontal

# Search for a horizontal line near the center
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line endpoints
        # Check if the line is nearly horizontal (y difference < 10) and close to center_y (within threshold)
        if abs(y1 - y2) < 10 and abs(y1 - center_y) < threshold:
            horizontal_line = (x1, y1, x2, y2)  # Set this as the horizontal line
            break

# If no suitable horizontal line is found, use the image center as a default
if horizontal_line is None:
    horizontal_line = (0, center_y, width, center_y)  # Default line spans image width at center_y

# Draw the horizontal line on the image in red
x1, y1, x2, y2 = horizontal_line
cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line with thickness 2
# Add label for the horizontal line
cv2.putText(image, "Horizontal Line (Center)", (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red text with scale 0.6

# Mark the center point with a green dot and label
cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)  # Green filled circle
cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green text with coordinates

# Convert image from BGR (OpenCV) to RGB (Matplotlib) for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define constants for pixel-to-mm conversion
KNOWN_OBJECT_SIZE_MM = 25  # Known size of a reference object in millimeters
KNOWN_OBJECT_SIZE_PIXELS = 242  # Known size of the reference object in pixels
PIXELS_PER_MM = KNOWN_OBJECT_SIZE_PIXELS / KNOWN_OBJECT_SIZE_MM  # Conversion factor (pixels/mm)

# Set up Matplotlib figure for interactive display
fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure with size 10x6 inches
ax.imshow(image_rgb)  # Display the RGB image
ax.set_title("Click to Measure Perpendicular Distance from Center Line")  # Add title
ax.axis("off")  # Hide axes for cleaner display

# Define the path for the CSV output file
csv_path = os.path.join(os.getcwd(), 'click_measurements.csv')  # Save in current working directory

# Initialize CSV file with headers
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "Diameter(X)"])  # Write CSV header

click_data = []  # List to store click data
label_counter = 1  # Counter for labeling clicked points (e.g., A1, A2, ...)

def onclick(event):
    """Handle mouse click events to measure perpendicular distance from the center line."""
    global label_counter  # Access global counter

    # Check if the click is within the image bounds
    if event.ydata is None or event.xdata is None:
        return

    # Get click coordinates (convert to integers)
    x_click, y_click = int(event.xdata), int(event.ydata)

    # Calculate perpendicular distance from the click point to the center line in pixels
    pixel_distance = center_y - y_click  # Assuming center_y is the reference line
    mm_distance = pixel_distance / PIXELS_PER_MM  # Convert to millimeters

    # Generate a label for the clicked point
    label = f"A{label_counter}"
    label_counter += 1  # Increment counter for the next label

    # Store click data
    click_data.append([x_click, y_click, mm_distance, label])

    # Draw a green dashed line from the click point to the center line
    ax.plot([x_click, x_click], [y_click, center_y], 'g--', linewidth=1)
    # Mark the click point with a blue dot
    ax.scatter(x_click, y_click, color='blue', marker='o')
    # Add label and distance text near the click point in blue
    ax.text(x_click + 10, y_click - 10, f"{label} ({mm_distance:.2f} mm)", color='blue', fontsize=8, weight='bold')

    # Update the figure display
    fig.canvas.draw()

    # Append the measurement to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([x_click, y_click, mm_distance, label])  # Write x, y, distance, and label

# Connect the click event handler to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

# Display the interactive plot
plt.show()

# Print the location where data is saved
print(f"Data saved to {csv_path}")