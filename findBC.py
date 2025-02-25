import math

def calculate_radius(center, points):
    """Calculate the radius of the circle given its center and a point on the circle."""
    x_c, y_c = center
    radii = [math.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) for x, y in points]
    
    # Since all points should have the same radius, we return the first one
    return radii[0] if radii else None

# Example usage
center = (3, 4)  # Example center of the circle
points = [(6, 8), (0, 4)]  # Example points on the circle

radius = calculate_radius(center, points)
print(f"Radius of the circle: {radius}")