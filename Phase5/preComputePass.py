from Phase4.backbone import TruncatedResNetBackbone, feature_to_pixel, normalizedImage
from Phase3.dataset import CustomDataset
import torch #type: ignore



datasetPath = "Phase3\\dataset\\seq-01\\seq-01"
dataset = CustomDataset(datasetPath)
backbone = TruncatedResNetBackbone()
samples_per_frame = 512

feature_list = []
pixelcords_list = []
frameindices_list = []
pose_list = []

for frame_idx in range(len(dataset)):

    img, pose = dataset[frame_idx]
    # print(img.shape)
    img_normalized = normalizedImage(img)

    # print(img_normalized.shape)

    with torch.no_grad():
        img_featured = backbone(img_normalized)
    # print(img_featured.shape)

    img_featured = img_featured.squeeze(0)
    _, grid_h, grid_w = img_featured.shape


    flat_indices = torch.randperm(grid_h*grid_w)[:samples_per_frame]
    # print( flat_indices.shape)
    rows, cols = torch.unravel_index(flat_indices, (grid_h, grid_w))
    # print(rows)
    # print(rows.shape, cols.shape)



    u,v = feature_to_pixel(rows,cols)
    stacked = torch.stack([u, v], dim=1)
    # print(stacked)
    pixelcords_list.append(stacked)

    pose_list.append(pose)

    sampled_features = img_featured[:, rows, cols]
    sampled_features = sampled_features.permute(1,0)
    feature_list.append(sampled_features)
    # print(sampled_features.shape)



    frame_index = torch.full((samples_per_frame,), frame_idx)
    # print(frame_index)
    frameindices_list.append(frame_index)
############################################################################

all_poses = torch.stack(pose_list, dim=0)
all_features = torch.cat(feature_list, dim=0)
all_frameIndices = torch.cat(frameindices_list, dim=0)
all_pixelcoords = torch.cat(pixelcords_list, dim=0)

seq_1_buffer = {
    "features": all_features,
    "pixel_coords": all_pixelcoords,
    "frame_indices": all_frameIndices,
    "poses": all_poses,
}
torch.save(seq_1_buffer, "seq_1_buffer.pt")


# print("all_poses shape: ", all_poses.shape)
# print("all_features shape: ", all_features.shape)
# print("all_frameIndices shape: ", all_frameIndices.shape)
# print("all_pixelcoords shape: ", all_pixelcoords.shape)
