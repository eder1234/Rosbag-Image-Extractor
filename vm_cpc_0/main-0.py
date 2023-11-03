#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree

def numpy_to_pointcloud(points_np, colors_np):
    """
    Convert numpy arrays of points and colors to an open3d PointCloud object.
    
    Parameters:
    - points_np (numpy.ndarray): The 3D points.
    - colors_np (numpy.ndarray): The associated colors.
    
    Returns:
    - o3d.geometry.PointCloud: The PointCloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np / 255.0)  # Normalize to [0,1]
    
    return pcd

def downsample_pointcloud_with_colors(pcd, colors, voxel_size=0.05):
    """
    Downsample the point cloud and associated colors using voxel grid downsampling.
    
    Parameters:
    - pcd (o3d.geometry.PointCloud): The input point cloud.
    - colors (numpy.ndarray): The associated colors.
    - voxel_size (float): Size of the voxel to use for downsampling.
    
    Returns:
    - tuple: The downsampled point cloud and downsampled colors.
    """
    down_pcd = pcd.voxel_down_sample(voxel_size)
    down_colors = np.asarray(down_pcd.colors) * 255
    down_colors = down_colors.astype(np.uint8)
    return down_pcd, down_colors

def compute_fpfh_color_descriptor_optimized(point_cloud, colors, radius=0.1, hist_bins=16):
    """
    Compute a combined FPFH and color histogram descriptor for the given point cloud. Optimized for speed.
    
    Parameters:
    - point_cloud (o3d.geometry.PointCloud): The point cloud to compute the descriptor for.
    - colors (numpy.ndarray): The colors associated with each point in the point cloud.
    - radius (float): The radius to consider for the local neighborhood of each point.
    - hist_bins (int): Number of bins to use for the color histogram.
    
    Returns:
    - numpy.ndarray: The combined FPFH + Color descriptor for each point in the point cloud.
    """
    
    print("Computing FPFH+Color")
    
    # Compute the FPFH descriptor
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(point_cloud, o3d.geometry.KDTreeSearchParamRadius(radius))
    
    # Compute color histograms for each point
    r_hist = np.histogram(colors[:, 0], bins=hist_bins, range=(0, 256))[0]
    g_hist = np.histogram(colors[:, 1], bins=hist_bins, range=(0, 256))[0]
    b_hist = np.histogram(colors[:, 2], bins=hist_bins, range=(0, 256))[0]
    color_histogram = np.concatenate([r_hist, g_hist, b_hist])
    
    # Replicate the color histogram for all points in point cloud
    color_histograms = np.tile(color_histogram, (len(point_cloud.points), 1))
    
    # Concatenate the FPFH descriptor and color histograms
    combined_descriptor = np.hstack([fpfh.data.T, color_histograms])
    
    return combined_descriptor


def load_image_pairs(color_folder, depth_folder):
    """
    Load image pairs from specified color and depth folders.
    Assumes that for each color image there's a corresponding depth image with the same name.
    """
    color_images = sorted([os.path.join(color_folder, f) for f in os.listdir(color_folder)])
    depth_images = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder)])
    
    return list(zip(color_images, depth_images))

def generate_colored_point_cloud(color_img_path, depth_img_path, color_camera_info, depth_camera_info):
    
    print("Generating point cloud")
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



def select_frame_optimized(current_cpc, reference_cpc, threshold=9326, query_fraction=0.1):
    """
    Optimized version of select_frame that uses a fraction of the descriptors for KDTree querying.
    
    Parameters:
    - current_cpc (tuple): Tuple of point cloud and colors for the current frame.
    - reference_cpc (tuple): Tuple of point cloud and colors for the reference frame.
    - threshold (float): The threshold to decide if the current frame should be selected.
    - query_fraction (float): Fraction of descriptors to be used for KDTree querying.
    
    Returns:
    - bool: True if the current frame should replace the reference frame, False otherwise.
    """
    print("Compute FPFH + Color")
    # Compute the combined descriptor for both point clouds
    current_descriptor = compute_fpfh_color_descriptor_optimized(current_cpc[0], current_cpc[1])
    reference_descriptor = compute_fpfh_color_descriptor_optimized(reference_cpc[0], reference_cpc[1])
    
    # Randomly sample a subset of descriptors for querying
    num_descriptors = current_descriptor.shape[0]
    num_query_descriptors = int(num_descriptors * query_fraction)
    query_indices = np.random.choice(num_descriptors, num_query_descriptors, replace=False)
    query_descriptors = current_descriptor[query_indices]
    
    print("cKDTree")
    # Use cKDTree for efficient nearest neighbor search
    tree = cKDTree(reference_descriptor)
    distances, _ = tree.query(query_descriptors, k=1)
    avg_distance = np.mean(distances)
    print("Average distance: ", avg_distance)
    
    return avg_distance > threshold

def estimate_normals(pcd, radius=0.1):
    """
    Estimate normals for the point cloud.
    
    Parameters:
    - pcd (o3d.geometry.PointCloud): The input point cloud.
    - radius (float): Search radius to use for normal estimation.
    
    Returns:
    - o3d.geometry.PointCloud: The point cloud with normals estimated.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd


def main_updated(color_folder, depth_folder):
    image_pairs = load_image_pairs(color_folder, depth_folder)
    reference_frame = image_pairs[0]
    
    # Camera Info (from your provided code)
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

    for current_frame in image_pairs[1:]:
        current_points, current_colors = generate_colored_point_cloud(current_frame[0], current_frame[1], color_camera_info, depth_camera_info)
        reference_points, reference_colors = generate_colored_point_cloud(reference_frame[0], reference_frame[1], color_camera_info, depth_camera_info)
    
        # Convert numpy arrays to PointCloud objects
        current_pcd = numpy_to_pointcloud(current_points, current_colors)
        reference_pcd = numpy_to_pointcloud(reference_points, reference_colors)
        
        # Downsample the point clouds and their colors
        voxel_size = 2 # Adjust as needed
        current_pcd_downsampled, current_colors_downsampled = downsample_pointcloud_with_colors(current_pcd, current_colors, voxel_size)
        reference_pcd_downsampled, reference_colors_downsampled = downsample_pointcloud_with_colors(reference_pcd, reference_colors, voxel_size)
        
        # Estimate normals for the downsampled point clouds
        current_pcd_downsampled = estimate_normals(current_pcd_downsampled)
        reference_pcd_downsampled = estimate_normals(reference_pcd_downsampled)
    
        if select_frame_optimized((current_pcd_downsampled, current_colors_downsampled), (reference_pcd_downsampled, reference_colors_downsampled)):
            reference_frame = current_frame
            print(f"New reference frame selected: {reference_frame[0]}")
        else:
            print("Frame not selected")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./main.py <path_to_color_images> <path_to_depth_images>")
        sys.exit(1)

    color_folder = sys.argv[1]
    depth_folder = sys.argv[2]
    main_updated(color_folder, depth_folder)

