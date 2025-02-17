import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

image_path = "/Users/kushagravarshney/Desktop/ultraProject/project/reverseSideView.jpeg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

height, width = gray.shape
center_x, center_y = width // 2, height // 2

horizontal_line = None
threshold = 15

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10 and abs(y1 - center_y) < threshold:
            horizontal_line = (x1, y1, x2, y2)
            break

if horizontal_line is None:
    horizontal_line = (0, center_y, width, center_y)

x1, y1, x2, y2 = horizontal_line
cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.putText(image, "Horizontal Line (Center)", (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

KNOWN_OBJECT_SIZE_MM = 25
KNOWN_OBJECT_SIZE_PIXELS = 242
PIXELS_PER_MM = KNOWN_OBJECT_SIZE_PIXELS / KNOWN_OBJECT_SIZE_MM

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_rgb)
ax.set_title("Click to Measure Perpendicular Distance from Center Line")
ax.axis("off")

csv_path = os.path.join(os.getcwd(), 'click_measurements.csv')

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "Diameter(X)"])

click_data = []
label_counter = 1


def onclick(event):
    global label_counter

    if event.ydata is None or event.xdata is None:
        return

    x_click, y_click = int(event.xdata), int(event.ydata)

    pixel_distance = center_y - y_click
    mm_distance = pixel_distance / PIXELS_PER_MM

    label = f"A{label_counter}"
    label_counter += 1

    click_data.append([x_click, y_click, mm_distance, label])

    ax.plot([x_click, x_click], [y_click, center_y], 'g--', linewidth=1)
    ax.scatter(x_click, y_click, color='blue', marker='o')
    ax.text(x_click + 10, y_click - 10, f"{label} ({mm_distance:.2f} mm)", color='blue', fontsize=8, weight='bold')

    fig.canvas.draw()

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([x_click, y_click, mm_distance, label])


fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print(f"Data saved to {csv_path}")