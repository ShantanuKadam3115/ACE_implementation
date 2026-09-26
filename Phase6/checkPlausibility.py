import torch #type: ignore
from torch.utils.data import DataLoader #type: ignore

from Phase5.bufferDataset import Bufferdataset
from Phase6.MLPCreation import MLP


if __name__ == "__main__":
    # load trained weights for pure inference
    model = MLP()
    model.load_state_dict(torch.load("Phase6/mlp_weights.pt"))
    model.eval()  # no BatchNorm/Dropout here, but keep the habit

    buffer_dataset = Bufferdataset(path="seq_1_buffer.pt")
    buffer_loader = DataLoader(buffer_dataset, batch_size=2048, shuffle=True)

    # any batch will do for a sanity check, there's no held-out split yet
    feature_batch, pixel_coords_batch, pose_batch = next(iter(buffer_loader))

    with torch.no_grad():
        predicted_world = model(feature_batch)  # (B, 3)

    # reference range: every camera position in the sequence, not just this batch
    poses = buffer_dataset.data["poses"]  # (1000, 4, 4), camera-to-world
    cam_positions = poses[:, :3, 3]       # (1000, 3)

    pred_min, pred_max = predicted_world.min(dim=0).values, predicted_world.max(dim=0).values
    cam_min, cam_max = cam_positions.min(dim=0).values, cam_positions.max(dim=0).values
    pred_mean, cam_mean = predicted_world.mean(dim=0), cam_positions.mean(dim=0)
    pred_std, cam_std = predicted_world.std(dim=0), cam_positions.std(dim=0)

    print(f"predicted points: {predicted_world.shape[0]}, camera poses: {cam_positions.shape[0]}\n")
    print(f"{'axis':<5}{'pred min':>10}{'pred max':>10}{'pred mean':>11}{'pred std':>10}"
          f"   |{'cam min':>10}{'cam max':>10}{'cam mean':>11}{'cam std':>10}")
    for i, axis in enumerate("xyz"):
        print(f"{axis:<5}{pred_min[i]:>10.3f}{pred_max[i]:>10.3f}{pred_mean[i]:>11.3f}{pred_std[i]:>10.3f}"
              f"   |{cam_min[i]:>10.3f}{cam_max[i]:>10.3f}{cam_mean[i]:>11.3f}{cam_std[i]:>10.3f}")

    # robust spread: the 1st/99th percentiles ignore a handful of wild outliers
    q = torch.tensor([0.01, 0.99])
    pred_q = torch.quantile(predicted_world, q, dim=0)  # (2, 3)
    print("\npredicted 1st/99th percentile per axis:")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: [{pred_q[0, i]:.3f}, {pred_q[1, i]:.3f}]")

    # distance from each predicted point to the nearest camera position
    dist_to_cam = torch.cdist(predicted_world, cam_positions).min(dim=1).values
    print(f"\ndistance to nearest camera (m): median {dist_to_cam.median():.3f}, "
          f"90th pct {torch.quantile(dist_to_cam, 0.9):.3f}, max {dist_to_cam.max():.3f}")
