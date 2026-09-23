import torch #type: ignore
import torch.nn as nn  #type: ignore
import torch.nn.functional as F  #type: ignore
from torch.utils.data import DataLoader #type: ignore

from Phase5.bufferDataset import Bufferdataset

fx, fy = 532.57, 531.54
cx, cy = 320.0, 240.0
k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

NUM_TRAINING_STEPS = 1000

TOLERANCE_START = 200.0  # pixels, loose early on
TOLERANCE_END = 2.0      # pixels, tight once the network is capable


def get_tolerance(step, total_steps, start=TOLERANCE_START, end=TOLERANCE_END):
    # exponential decay: shrinks fast early, levels off later
    progress = min(step / total_steps, 1.0)
    return start * (end / start) ** progress


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
    torch.manual_seed(0)
    linearModel = MLP()
    optimizer = torch.optim.Adam(linearModel.parameters(), lr=1e-3)
    epochs = 7
    counter = 0
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for step, (feature_batch, pixel_coords_batch, pose_batch) in enumerate(buffer_loader):
            if step >= NUM_TRAINING_STEPS:
                break

            predicted_world = linearModel(feature_batch)  # (B, 3)

            
            # print(pose_batch[0])
            # print(pose_batch[511])
            # print(pose_batch[512])
            # print(feature_batch.shape)
            # print(pixel_coords_batch.shape)
            # print(pose_batch.shape) # check why this shape is (2048,4,4)

            R_cw = pose_batch[:, 0:3, 0:3]  
            t_cw = pose_batch[:, 0:3, 3]    

            # print(R_cw.shape)

            R_wc = R_cw.mT

            # t_wc = - R_wc @ t_cw.T 
            # X_cam = predicted_world @ R_wc.T + t_wc

            t_wc = -torch.einsum('bij,bj->bi', R_wc, t_cw)
            X_cam = torch.einsum('bij,bj->bi', R_wc, predicted_world) + t_wc  # (B, 3)

            # print(t_wc.shape)
            # print(X_cam.shape)

            uvw = X_cam @ k.T  # (B, 3), shared K, no batching needed here
            # print(uvw.shape)
            predicted_pixels = uvw[:, :2] / uvw[:, 2:3]  # (B, 2)

            # print(predicted_pixels.shape)

            per_sample_error = torch.norm(predicted_pixels - pixel_coords_batch, dim=1)  

            valid_mask = X_cam[:, 2] > 0  
            valid_errors = per_sample_error[valid_mask]
            # print(valid_errors.shape)

            if valid_errors.numel() == 0:
                print(f"batch {counter}: no valid (in-front-of-camera) samples, skipping")
                continue

            tolerance = get_tolerance(counter, NUM_TRAINING_STEPS)
            counter+= 1
            # print( tolerance)
            loss_per_sample = torch.clamp(valid_errors - tolerance, min=0)
            # print(loss_per_sample.shape)
            loss = loss_per_sample.mean()
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"step {step}: loss = {loss.item():.4f}, "
                f"tolerance = {tolerance:.2f}, "
                f"valid = {valid_errors.numel()}/{per_sample_error.numel()}"
            )
