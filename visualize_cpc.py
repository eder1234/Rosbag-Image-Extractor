import open3d as o3d

def visualize_ply(ply_path):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Use the function
visualize_ply("output.ply")
