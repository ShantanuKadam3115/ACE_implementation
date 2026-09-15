import torch #type: ignore
import torch.nn as nn  #type: ignore
import torch.nn.functional as F  #type: ignore
from torch.utils.data import DataLoader #type: ignore

from Phase5.bufferDataset import Bufferdataset

fx, fy = 532.57, 531.54
cx, cy = 320.0, 240.0
k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

NUM_TRAINING_STEPS = 1


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 30)
        self.layer2 = nn.Linear(30, 24)
        self.layer3 = nn.Linear(24, 12)
        self.layer4 = nn.Linear(12, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


if __name__ == "__main__":
    buffer_dataset = Bufferdataset(path="seq_1_buffer.pt")
    buffer_loader = DataLoader(buffer_dataset, batch_size=2048, shuffle=True)

    linearModel = MLP()
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.SGD(linearModel.parameters(), lr=0.001)

    for step, (feature_batch, pixel_coords_batch, pose_batch) in enumerate(buffer_loader):
        if step >= NUM_TRAINING_STEPS:
            break

        predicted_world = linearModel(feature_batch)  # (B, 3)

        R_cw = pose_batch[:, 0:3, 0:3]  # (B, 3, 3), one rotation per sample
        t_cw = pose_batch[:, 0:3, 3]    # (B, 3), one translation per sample

        R_wc = R_cw.mT

        # t_wc = - R_wc @ t_cw.T 
        # X_cam = predicted_world @ R_wc.T + t_wc

        t_wc = -torch.einsum('bij,bj->bi', R_wc, t_cw)
        X_cam = torch.einsum('bij,bj->bi', R_wc, predicted_world) + t_wc  # (B, 3)

        print(t_wc.shape)
        print(X_cam.shape)

        uvw = X_cam @ k.T  # (B, 3), shared K, no batching needed here
        predicted_pixels = uvw[:, :2] / uvw[:, 2:3]  # (B, 2)

        loss = lossFunction(predicted_pixels, pixel_coords_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step}: loss = {loss.item():.4f}")
