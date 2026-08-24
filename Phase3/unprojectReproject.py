import torch #type: ignore
import numpy as np
import os
from PIL import Image

from dataset import CustomDataset

fx, fy  = 525.0, 525.0
cx, cy = 319.5,239.5
k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])

dataset_path = "Phase3\\dataset\\seq-01\\seq-01"

def LoadDepthMap(idx):
    padded = f"{idx:06d}"
    image_path = os.path.join(dataset_path,  f"frame-{padded}.depth.png")
    depth_tensor = torch.from_numpy(np.array(Image.open(image_path)))
    return depth_tensor


# --- Stage 1: load one frame's color, pose, and depth ---

dataset = CustomDataset(dataset_path)
img, pose = dataset[15]
depth_map = LoadDepthMap(15)

# --- Stage 2: pick a pixel with valid depth ---
u, v = 320, 240
print(f"chosen pixel: u={u}, v={v}, raw depth={depth_map[v][u].item()}mm")

# 7-Scenes depth is in mm -> convert to meters
Z = depth_map[v][u] / 1000
print(f"depth in meters: Z={Z.item():.4f}")

# --- Stage 3: unproject pixel + depth -> camera-coordinate point ---
# inverse of the pinhole equations: u = fx*X/Z + cx, v = fy*Y/Z + cy
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
X_cam = torch.stack([X, Y, Z])
print(f"X_cam (camera coords): {X_cam}")

# --- Stage 4: camera coords -> world coords, using pose (camera-to-world) ---
R_cw = pose[0:3, 0:3]
t_cw = pose[0:3, 3]
X_world = R_cw @ X_cam + t_cw
print(f"X_world (world coords): {X_world}")

# --- Stage 5: world coords -> camera coords (inverse pose direction) ---
R_wc = R_cw.mT
t_wc = -R_wc @ t_cw
X_cam2 = X_world @ R_wc.T + t_wc
print(f"X_cam2 (recovered camera coords): {X_cam2}")

# --- Stage 6: camera coords -> pixel, via K ---
u1, v1, w1 = k @ X_cam2
u1 = u1 / w1
v1 = v1 / w1
print(f"reprojected pixel: u1={u1.item():.4f}, v1={v1.item():.4f}")

# --- Stage 7: compare against the original pixel ---
error = torch.sqrt((u1 - u) ** 2 + (v1 - v) ** 2)
print(f"original pixel: u={u}, v={v}")
print(f"reprojection error: {error.item():.6f} pixels")

