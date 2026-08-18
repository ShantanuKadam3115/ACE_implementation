import torch #type: ignore
import numpy as np
import cv2

R_cw = torch.tensor([[0.7071,0.0,-0.7071],
                     [0.0,-1.0,0.0],
                     [-0.7071,0.0,-0.7071]])

R_cw_np = R_cw.numpy()

rvec,_ = cv2.Rodrigues(R_cw_np)

original_rcw, _ = cv2.Rodrigues(rvec)


theta = np.linalg.norm(rvec)
theta_deg = np.degrees(np.linalg.norm(rvec))
axis = rvec.flatten() / theta


print(rvec)
print(original_rcw)
print(theta)
print(theta_deg)
print(axis)