import torch #type: ignore
import torch.nn as nn  #type: ignore
import torch.nn.functional as F  #type: ignore
from torch.utils.data import DataLoader #type: ignore
import matplotlib.pyplot as plt #type: ignore

from Phase5.bufferDataset import Bufferdataset

fx, fy = 532.57, 531.54
cx, cy = 320.0, 240.0
k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

k_inv = torch.linalg.inv(k)

NUM_TRAINING_STEPS = 250

TOLERANCE_START = 400.0  # pixels, soft clamp early on (ACE default)
TOLERANCE_END = 1.0     # pixels, soft clamp once the network is capable

DEPTH_EPS = 1e-6         # only used to avoid dividing by zero when projecting
DEPTH_TARGET = 10.0      # metres, depth of the fallback target for invalid predictions


def get_tolerance(step, total_steps, start=TOLERANCE_START, end=TOLERANCE_END):
    # ACE's "circle" schedule: stays loose for a while, then drops quickly at the end
    progress = min(step / total_steps, 1.0)
    return end + (start - end) * (1.0 - progress) ** 0.5


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 120)
        self.layer3 = nn.Linear(120, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


if __name__ == "__main__":
    buffer_dataset = Bufferdataset(path="seq_1_buffer.pt")
    buffer_loader = DataLoader(buffer_dataset, batch_size=2048, shuffle=True)
    # torch.manual_seed(0)
    linearModel = MLP()
    optimizer = torch.optim.Adam(linearModel.parameters(), lr=1e-3)
    epochs = 16
    # tolerance decays over the whole run, not a fixed step count
    total_steps = epochs * min(len(buffer_loader), NUM_TRAINING_STEPS)
    counter = 0
    step_losses = []   # loss at every optimizer step (global)
    step_median_errors = []  # median reprojection error (px) of valid points per step
    step_valid_fracs = []    # fraction of points that are valid per step
    epoch_losses = []  # mean loss per epoch
    epoch_boundaries = []  # global step index where each epoch ends

    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        epoch_loss_sum, epoch_steps = 0.0, 0
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
            # clamp depth before dividing so points at/behind the camera don't produce inf/NaN
            depth = X_cam[:, 2:3]
            predicted_pixels = uvw[:, :2] / depth.clamp(min=DEPTH_EPS)  # (B, 2)
            # print("depth shape",depth.shape)
            # print(predicted_pixels.shape)

            per_sample_error = torch.norm(predicted_pixels - pixel_coords_batch, dim=1)
            # print("per_sample_error.shape ", per_sample_error.shape)

            depth = depth.squeeze(1)
            # print("depth", depth.shape)
            valid_mask = depth > 0  # only predictions behind the camera are invalid
            # print("valid_mask : ",valid_mask)
            invalid_mask = ~valid_mask
            # print("invalid mask : ", invalid_mask)
            valid_errors = per_sample_error[valid_mask]
            # print("valid errors shape : ",valid_errors.shape)

            tolerance = get_tolerance(counter, total_steps)
            counter+= 1
            # print( "tolerance: ", tolerance)

            # valid points: soft clamp, every point gets gradient, large errors are capped at ~tolerance
            # loss_valid = (tolerance * torch.tanh(valid_errors / tolerance)).sum()
            loss_valid = valid_errors.sum()
            # print("loss_valid: ", loss_valid)
            # invalid points: pull them towards a point DEPTH_TARGET metres along the pixel's ray
            loss_invalid = torch.zeros(())
            if invalid_mask.any():
                pixels_h = torch.cat(
                    [pixel_coords_batch[invalid_mask], torch.ones(int(invalid_mask.sum()), 1)], dim=1
                )  # (N, 3) homogeneous pixels
                target_cam = (pixels_h @ k_inv.T) * DEPTH_TARGET  # (N, 3) in camera frame
                target_world = (
                    torch.einsum('bij,bj->bi', R_cw[invalid_mask], target_cam) + t_cw[invalid_mask]
                )
                # du/dX ≈ fx/Z: at the target depth, 1 m of 3D error ≈ fx / DEPTH_TARGET pixels,
                # so scale the metre-space distance to be roughly commensurable with loss_valid (pixels)
                pixels_per_metre = k[0, 0] / DEPTH_TARGET
                loss_invalid = (
                    torch.norm(predicted_world[invalid_mask] - target_world, dim=1) * pixels_per_metre
                ).sum()
            # print("loss invalid: ", loss_invalid)
            loss = (loss_valid + loss_invalid) / per_sample_error.numel()
            # print("loss: ", loss)
            # print("per_sample_error.numel() : ",per_sample_error.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            median_error = valid_errors.median().item() if valid_errors.numel() > 0 else float("nan")
            valid_frac = valid_errors.numel() / per_sample_error.numel()

            step_losses.append(loss.item())
            step_median_errors.append(median_error)
            step_valid_fracs.append(valid_frac)
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            print(
                f"step {step}: loss = {loss.item():.4f}, "
                f"median err = {median_error:.1f}px, "
                f"tolerance = {tolerance:.2f}, "
                f"valid loss ={loss_valid:.2f}px,"
                f"valid = {valid_errors.numel()}/{per_sample_error.numel()}"
            )

        if epoch_steps > 0:
            epoch_losses.append(epoch_loss_sum / epoch_steps)
        epoch_boundaries.append(len(step_losses))


    torch.save(linearModel.state_dict(), "Phase6/mlp_weights.pt")
    
    # plot loss per step, loss per epoch, and the raw error / valid fraction
    fig, ((ax_step, ax_epoch), (ax_err, ax_valid)) = plt.subplots(2, 2, figsize=(12, 9))

    ax_step.plot(range(len(step_losses)), step_losses, linewidth=1)
    for b in epoch_boundaries[:-1]:
        ax_step.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    ax_step.set_xlabel("step (global)")
    ax_step.set_ylabel("loss")
    ax_step.set_title("Loss per step (dashed = epoch boundary)")
    ax_step.grid(alpha=0.3)

    ax_epoch.plot(range(len(epoch_losses)), epoch_losses, marker="o")
    ax_epoch.set_xlabel("epoch")
    ax_epoch.set_ylabel("mean loss")
    ax_epoch.set_title("Mean loss per epoch")
    ax_epoch.grid(alpha=0.3)

    ax_err.plot(range(len(step_median_errors)), step_median_errors, linewidth=1)
    for b in epoch_boundaries[:-1]:
        ax_err.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("step (global)")
    ax_err.set_ylabel("median reprojection error (px)")
    ax_err.set_title("Median error of valid points")
    ax_err.grid(alpha=0.3)

    ax_valid.plot(range(len(step_valid_fracs)), step_valid_fracs, linewidth=1)
    for b in epoch_boundaries[:-1]:
        ax_valid.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    ax_valid.set_ylim(0, 1.05)
    ax_valid.set_xlabel("step (global)")
    ax_valid.set_ylabel("valid fraction")
    ax_valid.set_title("Fraction of valid predictions")
    ax_valid.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("loss_curves.png", dpi=150)
    plt.show()
