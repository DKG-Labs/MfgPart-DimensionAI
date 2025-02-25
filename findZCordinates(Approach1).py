# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms

# # Load MiDaS model
# # model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# # model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
# model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
# model.eval()

# # Load image
# image_path = "/Users/kushagravarshney/Desktop/ultraProject/project/reverseSideView.jpeg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # # Preprocess image
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((384, 384)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR),  # Maintain aspect ratio
#     transforms.CenterCrop(384),  # Avoid distortion
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # MiDaS-specific normalization
# ])

# input_image = transform(image).unsqueeze(0)

# # Predict depth
# with torch.no_grad():
#     depth_map = model(input_image)

# # Example: If a reference object at (x_ref, y_ref) has real depth=1.5m
# ref_depth_real = 1.5  # meters
# ref_depth_pred = depth_map[y_ref, x_ref]  # Raw MiDaS output
# scaling_factor = ref_depth_real / ref_depth_pred
# calibrated_depth_map = depth_map * scaling_factor  # Now in real-world units

# # Convert depth map to numpy
# depth_map = depth_map.squeeze().cpu().numpy()

# # Normalize for visualization
# depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
# depth_map_vis = np.uint8(depth_map_vis)

# # Display depth map
# fig, ax = plt.subplots(figsize=(10, 6))
# cax = ax.imshow(depth_map_vis, cmap="plasma")
# fig.colorbar(cax, ax=ax)
# ax.set_title("Click to Get Z-Coordinate (Depth)")

# # Mouse click event to get Z-coordinate
# def onclick(event):
#     if event.xdata is None or event.ydata is None:
#         return

#     x_click, y_click = int(event.xdata), int(event.ydata)
    
#     # Get depth value at clicked point
#     z_coordinate = depth_map[y_click, x_click]
    
#     print(f"Clicked Point: ({x_click}, {y_click}) → Z-Coordinate: {z_coordinate:.2f}")

#     # Annotate the clicked point
#     ax.plot(x_click, y_click, "ro", markersize=5)
#     ax.text(x_click + 10, y_click, f"{z_coordinate:.2f}", color="white", fontsize=8, weight="bold")
    
#     fig.canvas.draw()

# # Connect click event
# fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()

# import cv2
# import numpy as np
# import open3d as o3d

# # Step 1: Load Images
# image1 = cv2.imread('/Users/kushagravarshney/Desktop/frontView.jpeg')
# image2 = cv2.imread('/Users/kushagravarshney/Desktop/reserveSide.jpeg')

# # Step 2: Convert to Grayscale
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# # Step 3: Feature Matching using SIFT with Ratio Test
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# # Use BFMatcher with k-NN and Lowe's ratio test
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# ratio_thresh = 0.85  # Increased threshold for more matches
# good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

# print(f"Number of good matches: {len(good_matches)}")
# if len(good_matches) < 8:
#     raise ValueError("Not enough good matches to proceed.")

# # Step 4: Get Points from Good Matches
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# # Step 5: Compute Camera Matrix based on Image Dimensions
# h, w = gray1.shape
# f = max(h, w)  # Approximate focal length based on image size
# K = np.array([[f, 0, w // 2],
#               [0, f, h // 2],
#               [0, 0, 1]], dtype=float)

# # Step 6: Compute Essential Matrix with RANSAC
# E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=5.0)

# # Filter points using Essential Matrix inliers (using > 0 for safety)
# pts1 = pts1[mask_e.ravel() > 0]
# pts2 = pts2[mask_e.ravel() > 0]

# print(f"Number of inliers after Essential Matrix estimation: {len(pts1)}")
# if len(pts1) < 8:
#     raise ValueError("Not enough inliers after Essential Matrix estimation.")

# # Step 7: Recover Camera Pose
# _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K)
# num_inliers_pose = int(mask_p.sum())  # This sum might be high due to 255 values per inlier.
# print(f"Number of inliers after Pose Recovery (mask sum): {num_inliers_pose}")

# # Apply Pose Recovery inliers mask (using > 0 to account for 255 values)
# pts1 = pts1[mask_p.ravel() > 0]
# pts2 = pts2[mask_p.ravel() > 0]

# if len(pts1) == 0 or len(pts2) == 0:
#     raise ValueError("No inliers left after Pose Recovery.")

# # Step 8: Triangulate Points and Visualize
# P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
# P2 = np.dot(K, np.hstack((R, t.reshape(3, 1))))

# # cv2.triangulatePoints expects points in 2xN shape
# pts1 = pts1.T.astype(np.float32)
# pts2 = pts2.T.astype(np.float32)

# pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
# pts3D = (pts4D[:3] / pts4D[3]).T

# # Step 9: Visualize the Point Cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts3D)
# o3d.visualization.draw_geometries([pcd])

import cv2
import numpy as np
import open3d as o3d

# Step 1: Load Images
image1 = cv2.imread('/Users/kushagravarshney/Desktop/frontView.jpeg')
image2 = cv2.imread('/Users/kushagravarshney/Desktop/reserveSide.jpeg')

# Step 2: Convert to Grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Step 3: Feature Matching using SIFT with Ratio Test
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

ratio_thresh = 0.85
good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

print(f"Number of good matches: {len(good_matches)}")
if len(good_matches) < 8:
    raise ValueError("Not enough good matches to proceed.")

# Step 4: Get Points from Good Matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Step 5: Compute Camera Matrix based on Image Dimensions
h, w = gray1.shape
f = max(h, w)  # Approximate focal length based on image size
K = np.array([[f, 0, w // 2],
              [0, f, h // 2],
              [0, 0, 1]], dtype=float)

# Step 6: Compute Essential Matrix with RANSAC
E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                 prob=0.999, threshold=5.0)

# Filter points using Essential Matrix inliers (using > 0 for safety)
pts1 = pts1[mask_e.ravel() > 0]
pts2 = pts2[mask_e.ravel() > 0]

print(f"Number of inliers after Essential Matrix estimation: {len(pts1)}")
if len(pts1) < 8:
    raise ValueError("Not enough inliers after Essential Matrix estimation.")

# Step 7: Recover Camera Pose
_, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K)
num_inliers_pose = int(mask_p.sum())  # inliers might be marked as 255
print(f"Number of inliers after Pose Recovery (mask sum): {num_inliers_pose}")

# Filter using pose mask (using > 0 to account for 255 values)
pts1 = pts1[mask_p.ravel() > 0]
pts2 = pts2[mask_p.ravel() > 0]

if len(pts1) == 0 or len(pts2) == 0:
    raise ValueError("No inliers left after Pose Recovery.")

# Step 8: Triangulate Points
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R, t.reshape(3, 1))))

pts1 = pts1.T.astype(np.float32)  # shape (2, N)
pts2 = pts2.T.astype(np.float32)  # shape (2, N)

pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
pts3D = (pts4D[:3] / pts4D[3]).T  # shape (N, 3)

print("\n--- 3D Points Coordinates ---")
for i, p in enumerate(pts3D):
    print(f"Point {i}: {p}")

# Step 9: Visualize the Point Cloud with Labels (Open3D >= 0.15)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3D)

try:
    # Initialize the Open3D GUI application
    o3d.visualization.gui.Application.instance.initialize()

    # Use the new O3DVisualizer that supports text labels
    vis = o3d.visualization.O3DVisualizer("3D Points with Labels", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("PointCloud", pcd)

    # Add a text label for each point.
    # (For many points, consider labeling only a subset to avoid clutter.)
    for i, p in enumerate(pts3D):
        label_text = f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
        vis.add_3d_label(p, label_text)

    # Call show() with a Boolean parameter and then run the GUI event loop
    vis.show(True)  # Pass True as required by the API.
    o3d.visualization.gui.Application.instance.run()

except AttributeError:
    print("\nYour Open3D version doesn't support 'O3DVisualizer.add_3d_label'. "
          "Please update to Open3D >= 0.15 or remove the labeling code.")
    # Fallback: Show without labels in the old viewer
    o3d.visualization.draw_geometries([pcd])

# import cv2
# import numpy as np
# import open3d as o3d

# # Step 1: Load Images
# image1 = cv2.imread('/Users/kushagravarshney/Desktop/frontView.jpeg')
# image2 = cv2.imread('/Users/kushagravarshney/Desktop/reserveSide.jpeg')

# # Step 2: Convert to Grayscale
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# # Step 3: Feature Matching using SIFT with Ratio Test
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# ratio_thresh = 0.85
# good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

# print(f"Number of good matches: {len(good_matches)}")
# if len(good_matches) < 8:
#     raise ValueError("Not enough good matches to proceed.")

# # Step 4: Get Points from Good Matches
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# # Step 5: Compute Camera Matrix based on Image Dimensions
# h, w = gray1.shape
# f = max(h, w)  # Approximate focal length based on image size
# K = np.array([[f, 0, w // 2],
#               [0, f, h // 2],
#               [0, 0, 1]], dtype=float)

# # Step 6: Compute Essential Matrix with RANSAC
# E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
#                                  prob=0.999, threshold=5.0)

# # Filter points using Essential Matrix inliers
# pts1 = pts1[mask_e.ravel() > 0]
# pts2 = pts2[mask_e.ravel() > 0]

# print(f"Number of inliers after Essential Matrix estimation: {len(pts1)}")
# if len(pts1) < 8:
#     raise ValueError("Not enough inliers after Essential Matrix estimation.")

# # Step 7: Recover Camera Pose
# _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K)
# num_inliers_pose = int(mask_p.sum())  # inliers might be marked as 255
# print(f"Number of inliers after Pose Recovery (mask sum): {num_inliers_pose}")

# # Filter using pose mask
# pts1 = pts1[mask_p.ravel() > 0]
# pts2 = pts2[mask_p.ravel() > 0]

# if len(pts1) == 0 or len(pts2) == 0:
#     raise ValueError("No inliers left after Pose Recovery.")

# # Step 8: Triangulate Points
# P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
# P2 = np.dot(K, np.hstack((R, t.reshape(3, 1))))

# pts1 = pts1.T.astype(np.float32)  # shape (2, N)
# pts2 = pts2.T.astype(np.float32)  # shape (2, N)

# pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
# pts3D = (pts4D[:3] / pts4D[3]).T  # shape (N, 3)

# print("\n--- 3D Points Coordinates ---")
# for i, p in enumerate(pts3D):
#     print(f"Point {i}: {p}")

# # Step 9: Visualize the Point Cloud with Labels (Open3D >= 0.15)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts3D)

# try:
#     # Initialize the Open3D GUI application
#     o3d.visualization.gui.Application.instance.initialize()

#     # Use the new O3DVisualizer that supports text labels
#     vis = o3d.visualization.O3DVisualizer("3D Points with Labels", 1024, 768)
#     vis.show_settings = True
#     vis.add_geometry("PointCloud", pcd)

#     # Add a text label for each point (be mindful of clutter if you have many points)
#     for i, p in enumerate(pts3D):
#         label_text = f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
#         vis.add_3d_label(p, label_text)

#     vis.show(True)  # Show the window
#     o3d.visualization.gui.Application.instance.run()

# except AttributeError:
#     print("\nYour Open3D version doesn't support 'O3DVisualizer.add_3d_label'. "
#           "Please update to Open3D >= 0.15 or remove the labeling code.")
#     # Fallback: Show without labels in the old viewer
#     o3d.visualization.draw_geometries([pcd])

# # ----------------------------------------------------------------------------
# # Step 10: Reflect (Reproject) 3D points onto the FIRST IMAGE (image1)
# # ----------------------------------------------------------------------------
# # We assume the first camera is at the origin (R1=I, t1=0).
# # Then, we project the same 3D points back into image1 to see where they fall.

# # 1) Build homogeneous coords of the 3D points.
# ones = np.ones((pts3D.shape[0], 1), dtype=np.float32)
# pts3D_h = np.hstack([pts3D, ones])  # shape (N,4)

# # 2) The first camera's extrinsics: [R|t] = [I|0].
# R1 = np.eye(3, dtype=np.float32)
# t1 = np.zeros((3,), dtype=np.float32)
# Rt1 = np.hstack([R1, t1.reshape(3, 1)])  # shape (3,4)

# # 3) Project onto image1:  (3,N) = K * [R|t] * (4,N)
# proj_2D_h = (K @ Rt1 @ pts3D_h.T)  # shape (3, N)
# proj_2D = proj_2D_h[:2] / proj_2D_h[2]  # shape (2, N)

# # 4) Draw these reprojected points on a copy of image1
# image2_with_pts = image2.copy()
# for i in range(proj_2D.shape[1]):
#     x = int(proj_2D[0, i])
#     y = int(proj_2D[1, i])
#     # Draw a small green circle at each point
#     cv2.circle(image2_with_pts, (x, y), 5, (0, 255, 0), -1)

# # 5) Display or save the image
# cv2.imshow("Image 1 with Reprojected 3D Points", image2_with_pts)
# cv2.waitKey(0)
# cv2.destroyAllWindows()