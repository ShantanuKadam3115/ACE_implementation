import torch.nn as nn # type: ignore
import torch # type: ignore


# CameraPoint = 10 * torch.rand(4, 3)



CameraPoint = torch.tensor([
    [1.0, 2.0, 5.0],   # X = 1
    [3.0, 2.0, 5.0],   # X = 3
    [6.0, 3.0, 7.0],
    [6.0, 3.0, 4.0]
])

print(CameraPoint)


fx, fy  = 500.0, 500.0
cx, cy = 320.0,240.0
k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])

# print(k)

u,v, w = k @ CameraPoint.T



u = u/w
v = v/w




print(u,v)

    
