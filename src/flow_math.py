import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def compute_dense_flow(prev_gray, curr_gray):
    # Farneback dense optical flow
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def estimate_epipolar_inliers(flow, sample_step=8, ransac_thresh=1.5):
    h, w = flow.shape[:2]

    # Sample a grid of points to avoid running RANSAC on every single pixel
    y_coords, x_coords = np.mgrid[0:h:sample_step, 0:w:sample_step].reshape(2, -1)
    pts1 = np.vstack((x_coords, y_coords)).T.astype(np.float32)

    # Calculate tracked point locations from flow vectors
    u = flow[y_coords, x_coords, 0]
    v = flow[y_coords, x_coords, 1]
    pts2 = pts1 + np.vstack((u, v)).T

    # Find fundamental matrix to isolate background pixels obeying epipolar geometry
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, 0.99)

    inlier_grid = np.zeros((h, w), dtype=bool)
    if mask is not None:
        valid = mask.ravel() == 1
        inlier_grid[y_coords[valid], x_coords[valid]] = True

        # Dilate so sampled inliers cover local neighborhoods
        kernel = np.ones((sample_step, sample_step), np.uint8)
        inlier_grid = cv2.dilate(inlier_grid.astype(np.uint8), kernel).astype(bool)
    else:
        # Fallback if RANSAC fails to converge
        inlier_grid[:] = True

    return inlier_grid, F


def isolate_residual_motion_pca(flow, inlier_mask, n_components=2):
    h, w, _ = flow.shape
    W = flow.reshape(-1, 2)
    inliers = W[inlier_mask.reshape(-1)]

    # Safety check if too few points were kept
    if len(inliers) < n_components:
        inliers = W

    # Fit PCA on background motion to model camera ego-motion
    pca = PCA(n_components=min(n_components, inliers.shape[1]))
    pca.fit(inliers)

    # Reconstruct expected background flow and subtract it
    W_bg = pca.inverse_transform(pca.transform(W))
    flow_bg = W_bg.reshape(h, w, 2)

    residual = flow - flow_bg
    residual_mag = np.linalg.norm(residual, axis=2)

    return residual_mag, flow_bg


def extract_moving_clusters(residual_mag, motion_threshold=2.2, max_clusters=4):
    h, w = residual_mag.shape
    binary_mask = (residual_mag > motion_threshold).astype(np.uint8)

    # Clean isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

    y_indices, x_indices = np.where(binary_clean > 0)

    # Return early if not enough active pixels for an obstacle
    if len(x_indices) < 60:
        return [], binary_clean

    features = np.column_stack((x_indices, y_indices))
    k = min(max_clusters, max(1, len(features) // 600))

    if k == 1:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        return [(x_min, y_min, x_max - x_min, y_max - y_min)], binary_clean

    # Cluster spatial residual points into discrete objects
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(features)

    boxes = []
    for c in range(k):
        cluster_pts = features[labels == c]
        if len(cluster_pts) > 100:
            x_min, y_min = np.min(cluster_pts, axis=0)
            x_max, y_max = np.max(cluster_pts, axis=0)
            boxes.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

    return boxes, binary_clean
