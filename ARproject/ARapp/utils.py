import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

def guided_filter(guide_u8, src, r=4, eps=50.0):
    I = guide_u8.astype(np.float32)
    p = src.astype(np.float32)
    mean_I  = cv2.boxFilter(I,     cv2.CV_32F, (r, r))
    mean_p  = cv2.boxFilter(p,     cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    return mean_a * I + mean_b

def fill_small_holes(depth, max_hole_px=200):
    mask_invalid = (depth == 0).astype(np.uint8)
    if mask_invalid.sum() == 0: return depth
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_invalid, connectivity=8)
    filled = depth.copy()
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_hole_px: continue
        hole_mask = (labels == label_id).astype(np.uint8)
        # 简化版填充逻辑，保证后端性能
        valid_mask = (filled > 0) & (hole_mask == 0)
        if not valid_mask.any(): continue
        # 使用简单的均值或近邻填充（此处为演示，保持逻辑一致）
        filled[hole_mask > 0] = np.median(filled[valid_mask])
    return filled

def _fill_all_holes_for_filter(d):
    if (d > 0).all(): return d
    result = d.copy()
    invalid = result == 0
    _, nearest_idx = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    result[invalid] = d[nearest_idx[0][invalid], nearest_idx[1][invalid]]
    return result