import torch # type: ignore
import math
import cv2

R_cw = torch.tensor([[0.7071,0.0,-0.7071],
                     [0.0,-1.0,0.0],
                     [-0.7071,0.0,-0.7071]])

t_cw = torch.tensor([6.0,0.0,6.0])

#camera settings
fx, fy  = 500.0, 500.0
cx, cy = 320.0,240.0

k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])


world_points = torch.tensor([
    [5.0,  0.0, 5.0],
    [4.0,  0.0, 4.0],
    [3.0,  0.0, 3.0],
    [2.0,  0.0, 2.0],
    [1.0,  0.0, 1.0],

    [5.0,  1.0, 4.0],
    [4.0,  1.0, 3.0],
    [3.0,  1.0, 2.0],
    [2.0,  1.0, 1.0],
    [1.0,  1.0, 0.0],

    [5.0, -1.0, 4.0],
    [4.0, -1.0, 3.0],
    [3.0, -1.0, 2.0],
    [2.0, -1.0, 1.0],
    [1.0, -1.0, 0.0],

    [4.0,  2.0, 5.0],
    [3.0,  2.0, 4.0],
    [2.0,  2.0, 3.0],
    [4.0, -2.0, 5.0],
    [2.0, -2.0, 3.0],
])


R_wc = R_cw.mT
t_wc = - R_wc @ t_cw 
print("t_wc", t_wc)



X_cam = torch.empty(len(world_points), 3)




X_cam = world_points @ R_wc.T + t_wc

u1,v1, w1 = k @ X_cam.T
u1 = u1/w1
v1 = v1/w1

imagepoints = torch.stack([u1, v1], dim=1)

# print(f"x_cam = {X_cam}")
# print("u1: ",u1," v1: ",v1)

success ,rvec,tvec,_ = cv2.solvePnPRansac(world_points.numpy(),imagePoints=imagepoints.numpy(),cameraMatrix= k.numpy(), distCoeffs=None)

print(tvec)

original_r_cw, _ = cv2.Rodrigues(rvec)

# print("recovered rcw", original_r_cw.mT)

