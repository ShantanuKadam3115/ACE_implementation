import torch # type: ignore
import math

R_cw = torch.tensor([[0.7071,0.0,-0.7071],
                     [0.0,-1.0,0.0],
                     [-0.7071,0.0,-0.7071]])

t_cw = torch.tensor([2.0,0.0,2.0])

#camera settings
fx, fy  = 500.0, 500.0
cx, cy = 320.0,240.0
k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])

R_wc = R_cw.mT
t_wc = - R_wc @ t_cw.T #[0,0,5]


# true_wp = torch.tensor([-50.0, 50.0, -50.0])
true_wp = torch.tensor([0.0, 1.0, 0.0])


X_cam = torch.empty(len(true_wp), 3)
X_cam = true_wp @ R_wc.T + t_wc

u1,v1, w1 = k @ X_cam.T
u1 = u1/w1
v1 = v1/w1
print(f"x_cam = {X_cam}")
print("u1: ",u1," v1: ",v1)

# predicted_wp = torch.tensor([-50.01, 49.99, -50.001])
predicted_wp = torch.tensor([0.01, 0.99, 0.001])
predicted_cam = torch.empty(len(predicted_wp),3)
predicted_cam = predicted_wp @R_wc.T + t_wc
u2,v2, w2 = k @ predicted_cam.T
u2 = u2/w2
v2 = v2/w2

print(f"predicted_cam = {predicted_cam}")
print("u2: ",u2," v2: ",v2)

reprojection_error = math.dist((u1, v1), (u2, v2))

print(f"reprojection error is {reprojection_error}")