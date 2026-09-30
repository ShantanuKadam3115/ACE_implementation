from Phase4.backbone import TruncatedResNetBackbone , normalizedImage , feature_to_pixel
from Phase6.MLPCreation import MLP
from Phase3.dataset import CustomDataset
import torch  #type: ignore
import numpy as np
import cv2

fx, fy = 532.57, 531.54
cx, cy = 320.0, 240.0
K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

datasetPath = "Phase7\\testdata"
model = MLP()
model.load_state_dict(torch.load("Phase6/mlp_weights.pt"))
model.eval()

backbone = TruncatedResNetBackbone()
test_dataset = CustomDataset(datasetPath)

translation_error_list = []
rotation_error_list = []

for i in range(20):

    img, pose = test_dataset[i]
    img_normalized = normalizedImage(img)

    # print(pose.shape)
    true_R_cw = pose[0:3, 0:3]
    true_t_cw = pose[0:3, 3]

    with torch.no_grad():
        img_featured = backbone(img_normalized)

    img_featured = img_featured.squeeze(0)
    _, grid_h, grid_w = img_featured.shape
    # print(grid_h,grid_w)

    flat_indices = torch.arange(grid_h*grid_w)
    # print( flat_indices.shape)
    rows, cols = torch.unravel_index(flat_indices, (grid_h, grid_w))
    # print(rows.max())

    sampled_features = img_featured[:, rows, cols]
    sampled_features = sampled_features.permute(1,0)
    # print(sampled_features.shape)

    u,v = feature_to_pixel(rows,cols)
    pixels = torch.stack([u, v], dim=1)
    # print(pixels.shape)

    with torch.no_grad():
        predicted_world_points = model(sampled_features)

    # print(predicted_world_points.shape)

    
    # true_R_wc = true_R_cw.mT
    # true_t_wc = - true_R_wc @ true_t_cw 
    # # print("t_wc", t_wc)

    # check_X_cam = torch.empty(len(predicted_world_points), 3)
    # check_X_cam = predicted_world_points @ true_R_wc.T + true_t_wc

    # u1,v1, w1 = K @ check_X_cam.T
    # u1 = u1/w1
    # v1 = v1/w1
    # predicted_pixels = torch.stack([u1,v1], dim = 1)

    # # print(f"true pixels : {pixels} and predicted pixels ; {predicted_pixels}")
    # per_sample_error = torch.norm(predicted_pixels - pixels, dim=1)
    # total_error = per_sample_error.sum()
    # median_error = per_sample_error.median()
    # # print(median_error)


    success ,rvec,tvec,_ = cv2.solvePnPRansac(objectPoints=predicted_world_points.numpy(), imagePoints=pixels.numpy(), cameraMatrix=K.numpy(), distCoeffs=None,reprojectionError=245.0)

    # print(rvec, tvec)

    predicted_R_wc, _ = cv2.Rodrigues(rvec)
    predicted_t_wc = tvec.reshape(3)

    predicted_R_cw = predicted_R_wc.T                
    predicted_t_cw = -predicted_R_wc.T @ predicted_t_wc

    true_R_cw = true_R_cw.numpy().astype(np.float64)
    true_t_cw = true_t_cw.numpy().astype(np.float64)

    # translation error: distance between estimated and true camera centres (meters)
    translation_error = np.linalg.norm(predicted_t_cw - true_t_cw)
    translation_error_list.append(translation_error)

    # rotation error: angle of the relative rotation, from trace(R) = 1 + 2cos(theta)
    R_delta = predicted_R_cw.T @ true_R_cw
    cos_theta = (np.trace(R_delta) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # float drift can push it just outside [-1, 1]
    rotation_error = np.degrees(np.arccos(cos_theta))
    print(f"{i} : {rotation_error}")
    rotation_error_list.append(rotation_error)

print(f"success: {success}")
print(f"translation error list: {len(translation_error_list)} ")
print(f"rotation error list: {len(rotation_error_list)} ")
# print(translation_error_list)
# print(rotation_error_list)


median_rotation_error = np.median(rotation_error_list)
median_translation_error = np.median(translation_error_list)

print(f"Median rotation error: {median_rotation_error:.4f} deg")
print(f"Median translation error: {median_translation_error:.4f}")

