import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import math
from tqdm import tqdm
from collections import defaultdict

"""Loading and visualization functions
"""


def load_and_decimate_room_pc_voxels(path, voxel_size=0.10):
    """
    Loads a point cloud from a file and applies voxel downsampling.

    Parameters:
        path (str): File path to the point cloud.
        voxel_size (float): Size of the voxel used to downsample.

    Returns:
        tuple: (N, 3) array of point coordinates and (N, 3) array of point colors.
    """
    pcd = o3d.io.read_point_cloud(path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd_down.points)
    cols = np.asarray(pcd_down.colors)
    return pts, cols


def point_cloud_visu(pts, cls=None):
    """
    Displays a 3D scatter plot of the point cloud using Plotly.

    Parameters:
        pts (np.ndarray): (N, 3) array of point coordinates.
        cls (np.ndarray or None): Optional array of scalar values to color the points.
    """
    fig = go.Figure(
        data=[go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=cls)
        )],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                aspectratio=dict(x=1, y=1, z=0.95)
            )
        )
    )
    fig.show()


"""Point cloud transformation fonctions to get a 2D representation of a scene
"""


def plane_fitting_pca(pts):
    """
    Estimates a best-fit plane using PCA on a set of points.

    Parameters:
        pts (np.ndarray): (N, 3) array of points.

    Returns:
        np.ndarray: Plane parameters [a, b, c, d] where ax + by + cz + d = 0.
    """
    cov = np.cov(pts, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, pts.mean(axis=0))
    return np.append(normal, d)


def plane_from_three_points(pts):
    """
    Randomly selects three points and fits a plane through them using PCA.

    Parameters:
        pts (np.ndarray): (N, 3) array of points.

    Returns:
        np.ndarray: Plane parameters [a, b, c, d].
    """
    ids = np.random.choice(pts.shape[0], 3, replace=False)
    return plane_fitting_pca(pts[ids])


def get_inliers_in_plane(pts, plane, distance_threshold):
    """
    Returns a boolean mask of points close to the given plane.

    Parameters:
        pts (np.ndarray): Points to test.
        plane (np.ndarray): Plane parameters.
        distance_threshold (float): Max distance to consider as inlier.

    Returns:
        np.ndarray: Boolean mask of inliers.
    """
    distances = np.abs(pts @ plane[:3] + plane[3])
    return distances < distance_threshold


def normal_estimation(pts, k=16):
    """
    Estimates surface normals for each point using PCA on its k nearest neighbors.

    Parameters:
        pts (np.ndarray): (N, 3) point array.
        k (int): Number of neighbors for PCA.

    Returns:
        np.ndarray: (N, 3) array of normals.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    normals = []

    for i in tqdm(range(len(pts)), desc="Estimating normals", ncols=100):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighborhood = np.asarray(pcd.points)[idx]
        normal = plane_fitting_pca(neighborhood)[:3]
        normals.append(normal)

    return np.stack(normals)


def plane_from_one_point_normal(pts, normals):
    """
    Constructs a plane from a random point and its corresponding normal.

    Parameters:
        pts (np.ndarray): (N, 3) points.
        normals (np.ndarray): (N, 3) normals.

    Returns:
        np.ndarray: Plane parameters.
    """
    i = np.random.randint(len(pts))
    p, n = pts[i], normals[i]
    d = -np.dot(p, n)
    return np.append(n, d)


def get_inliers_in_plane_normals(normals, plane, orient_threshold):
    """
    Filters points by checking alignment of their normals with the plane's normal.

    Parameters:
        normals (np.ndarray): Normals of the points.
        plane (np.ndarray): Plane parameters.
        orient_threshold (float): Threshold on dot product deviation.

    Returns:
        np.ndarray: Boolean mask where orientation is consistent.
    """
    return np.abs(normals @ plane[:3]) > orient_threshold


def get_max_num_iter_normal(min_pts, num_pts, proba):
    """
    Estimates the number of RANSAC iterations needed to find a valid plane.

    Parameters:
        min_pts (int): Minimum number of points to define a plane.
        num_pts (int): Total number of points.
        proba (float): Desired success probability.

    Returns:
        int: Number of RANSAC iterations.
    """
    Pn = min_pts / num_pts
    return int(math.log(1 - proba) / math.log(1 - Pn)) + 1


def search_one_plane_normals(pts, normals, min_pts, plane_threshold, orient_threshold, proba_of_success):
    """
    Performs RANSAC to find one dominant plane aligned with X/Y/Z axes.

    Parameters:
        pts (np.ndarray): Points.
        normals (np.ndarray): Normals.
        min_pts (int): Minimum number of inliers.
        plane_threshold (float): Distance threshold for inliers.
        orient_threshold (float): Normal orientation threshold.
        proba_of_success (float): Probability of finding a good plane.

    Returns:
        tuple: Best plane and inlier mask.
    """
    best_plane, best_mask = None, None
    max_inliers = 0
    for _ in tqdm(range(get_max_num_iter_normal(min_pts, len(pts), proba_of_success)), ncols=100):
        plane = plane_from_one_point_normal(pts, normals)
        normal_plane = plane[:3] / np.linalg.norm(plane[:3])
        dot_z = np.abs(np.dot(normal_plane, np.array([0, 0, 1])))
        # Keep only horizontal and vertical planes (aligned with Z)
        if not (dot_z > 0.97 or dot_z < 0.03):
            continue
        mask_dist = get_inliers_in_plane(pts, plane, plane_threshold)
        mask_orient = get_inliers_in_plane_normals(
            normals, plane, orient_threshold)
        mask = np.logical_and(mask_dist, mask_orient)
        if mask.sum() > max_inliers:
            best_plane, best_mask = plane, mask
            max_inliers = mask.sum()
    return best_plane, best_mask


def ransac(pts_, normals_, min_pts, plane_search_function, plane_search_function_args):
    """
    Iteratively finds multiple planes in a point cloud using RANSAC.

    Parameters:
        pts_ (np.ndarray): Original points.
        normals_ (np.ndarray): Corresponding normals.
        min_pts (int): Minimum inliers for a valid plane.
        plane_search_function (function): Function to find one plane.
        plane_search_function_args (dict): Arguments for that function.

    Returns:
        list: List of arrays of inlier points for each detected plane.
    """
    pts = pts_.copy()
    normals = normals_.copy()
    shapes = []
    while True:
        plane, mask = plane_search_function(
            pts, normals, min_pts, **plane_search_function_args)
        if mask.sum() < min_pts:
            break
        print(f"Found one plane with {mask.sum()} inliers")
        shapes.append(pts[mask])
        pts = pts[~mask]
        normals = normals[~mask]
    return shapes


def density_img(x, y):
    """
    Creates a 2D image where each pixel intensity reflects the number of 3D points
    projected onto that (x, y) bin.

    Parameters:
        x, y (np.ndarray): Discretized voxel indices in X and Y.

    Returns:
        np.ndarray: 2D image (density map) normalized to [0, 1].
    """
    density_map = defaultdict(int)
    for xi, yi in zip(x, y):
        density_map[(xi, yi)] += 1

    map_width = x.max() + 1
    map_height = y.max() + 1
    img = np.zeros((map_height, map_width))

    for (xi, yi), count in density_map.items():
        img[yi, xi] = count

    img = np.clip(img / img.max(), 0, 1)
    return img


def height_map(x, y, z):
    """
    Creates a 2D image where each pixel contains the maximum Z (height)
    of all 3D points projected to that (x, y) location.

    Parameters:
        x, y (np.ndarray): Discretized voxel indices.
        z (np.ndarray): Corresponding Z (height) values.

    Returns:
        np.ndarray: Normalized height map (2D image).
    """
    height_map = {}
    map_width = x.max() + 1
    map_height = y.max() + 1
    for xi, yi, zi in zip(x, y, z):
        key = (xi, yi)
        if key not in height_map or zi > height_map[key]:
            height_map[key] = zi

    img_height = np.zeros((map_height, map_width))
    for (xi, yi), height in height_map.items():
        img_height[yi, xi] = height

    img_height = (img_height - img_height.min()) / \
        (img_height.max() - img_height.min())
    return img_height
