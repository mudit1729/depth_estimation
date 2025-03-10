#!/usr/bin/env python
import cv2
import numpy as np
import open3d as o3d
import argparse

def compute_depth_map(left_img, right_img, focal_length, baseline):
    # Convert images to grayscale for stereo matching
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Set stereo matcher parameters
    min_disp = 0
    num_disp = 16 * 6  # must be divisible by 16
    block_size = 7     # matching block size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity (result is scaled by 16)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Replace zero/negative disparities to avoid division by zero
    disparity[disparity <= 0.0] = 0.1

    # Compute depth: depth = (focal_length * baseline) / disparity
    depth = (focal_length * baseline) / disparity
    return depth, disparity

def create_point_cloud(depth, left_img, fx, fy, cx, cy, max_depth=100):
    h, w = depth.shape
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()

    # Convert left image from BGR to RGB and flatten
    color_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    colors = color_img.reshape(-1, 3) / 255.0

    # Filter out points with depth > max_depth to remove far outliers
    mask = depth_flat < max_depth
    u = u[mask]
    v = v[mask]
    depth_flat = depth_flat[mask]
    colors = colors[mask]
    
    # Reproject pixels to 3D using the pinhole camera model.
    # Note: We flip the y coordinate so that y is up (camera coordinate frame)
    X = (u - cx) * depth_flat / fx
    Y = -(v - cy) * depth_flat / fy  # flipped to have y up
    Z = depth_flat
    points = np.stack((X, Y, Z), axis=1)
    return points, colors

def visualize_point_cloud(pcd):
    # Create a custom visualizer to set the initial view
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Get the view control to modify camera parameters
    ctr = vis.get_view_control()
    
    # Set the view so that it mimics the camera coordinate frame:
    # - The camera (virtual) is placed so that its origin corresponds to the original camera's center.
    # - It is looking straight along the positive z-axis (i.e. front vector is [0, 0, 1]).
    # - The "lookat" is set to a point directly in front (e.g., [0, 0, 5]) so the scene appears as captured.
    # - The up vector is set as [0, 1, 0].
    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 5])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)
    
    print("Press 'q' or close the window to exit the visualization.")
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(
        description="3D visualization of a depth map as seen from the camera frame, with a colored point cloud from KITTI stereo images."
    )
    parser.add_argument('--left', type=str, required=True,
                        help="Path to the left image (from image_2)")
    parser.add_argument('--right', type=str, required=True,
                        help="Path to the right image (from image_3)")
    parser.add_argument('--focal', type=float, default=721.5377,
                        help="Focal length in pixels (default KITTI: 721.5377)")
    parser.add_argument('--baseline', type=float, default=0.54,
                        help="Baseline in meters (default KITTI: 0.54)")
    parser.add_argument('--cx', type=float, default=609.5593,
                        help="Principal point x-coordinate (default KITTI: 609.5593)")
    parser.add_argument('--cy', type=float, default=172.854,
                        help="Principal point y-coordinate (default KITTI: 172.854)")
    parser.add_argument('--max_depth', type=float, default=100,
                        help="Maximum depth to visualize (default: 100 meters)")
    args = parser.parse_args()

    # Load stereo images
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is None or right_img is None:
        raise ValueError("Could not load one or both images. Check the file paths.")

    # Compute depth map and disparity
    depth, disparity = compute_depth_map(left_img, right_img, args.focal, args.baseline)
    
    # Create a colored point cloud (points are in the camera coordinate frame)
    points, colors = create_point_cloud(depth, left_img, args.focal, args.focal, args.cx, args.cy, max_depth=args.max_depth)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Visualizing 3D point cloud in the camera frame...")
    visualize_point_cloud(pcd)

if __name__ == '__main__':
    main()
