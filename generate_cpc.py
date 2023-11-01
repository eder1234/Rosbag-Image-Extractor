import numpy as np
import cv2

def generate_colored_point_cloud(color_img_path, depth_img_path, color_camera_info, depth_camera_info):
    # Load the color and depth images
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

    # Extract camera matrices and distortion coefficients
    K_color = np.array(color_camera_info["K"])
    D_color = np.array(color_camera_info["D"])

    K_depth = np.array(depth_camera_info["K"])
    D_depth = np.array(depth_camera_info["D"])

    # Undistort images
    color_img_undistorted = cv2.undistort(color_img, K_color, D_color)
    depth_img_undistorted = cv2.undistort(depth_img, K_depth, D_depth)

    # Generate 3D points from depth image
    points = []
    colors = []
    for v in range(depth_img_undistorted.shape[0]):
        for u in range(depth_img_undistorted.shape[1]):
            depth = depth_img_undistorted[v, u]
            if depth == 0:  # Skip if no depth
                continue

            # Project (u, v, depth) to 3D space
            x = (u - K_depth[0, 2]) * depth / K_depth[0, 0]
            y = (v - K_depth[1, 2]) * depth / K_depth[1, 1]
            z = depth

            # Get color from color image
            color = color_img_undistorted[v, u]
            colors.append(color)
            points.append([x, y, z])

    # Convert to numpy arrays
    points_np = np.array(points)
    colors_np = np.array(colors)

    return points_np, colors_np

def save_point_cloud_to_ply(points, colors, filename):
    # Prepare header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]

    # Write to file
    with open(filename, 'w') as ply_file:
        # Write header
        for line in header:
            ply_file.write(line + "\n")

        # Write point cloud data
        for point, color in zip(points, colors):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {color[2]} {color[1]} {color[0]}\n")

# Use the function
# save_point_cloud_to_ply(points, colors, "output.ply")


# Camera Info
color_camera_info = {
    "K": [
        [929.994628906, 0.0, 643.123168945],
        [0.0, 929.571960449, 356.984924316],
        [0.0, 0.0, 1.0]
    ],
    "D": [0.0, 0.0, 0.0, 0.0, 0.0]
}

depth_camera_info = {
    "K": [
        [937.661193848, 0.0, 612.535644531],
        [0.0, 937.661193848, 364.03213501],
        [0.0, 0.0, 1.0]
    ],
    "D": [0.0, 0.0, 0.0, 0.0, 0.0]
}

# Generate the point cloud
points, colors = generate_colored_point_cloud('color_1000.png', 'depth_1000.png', color_camera_info, depth_camera_info)
save_point_cloud_to_ply(points, colors, "output.ply")