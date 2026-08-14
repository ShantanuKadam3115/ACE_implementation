import torch # type: ignore

R_cw = torch.tensor([[0.7071,0.0,0.7071],
                     [0.0,1.0,0.0],
                     [-0.7071,0.0,0.7071]])

t_cw = torch.tensor([2.0,0.0,2.0])

world_points = torch.tensor([[2.0,0.0,2.0],
                            [1.0,2.0,1.0],
                            [-1.0,2.0,1.0],
                            [-1.0,-2.0,1.0,],
                            [1.0,-2.0,1.0],
                            [4.0,0.0,4.0],
                            [0.0,1.0,0.0]])

R_wc = R_cw.mT
t_wc = - R_wc @ t_cw.T #[0,0,5]



X_cam = torch.empty(len(world_points), 3)



X_cam = world_points @ R_wc.T + t_wc


print(X_cam)