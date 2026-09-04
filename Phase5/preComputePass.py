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

    img_normalized = normalizedImage(img)

    with torch.no_grad():
        img_featured = backbone(img_normalized)

    img_featured = img_featured.squeeze(0)
    _, grid_h, grid_w = img_featured.shape


    flat_indices = torch.randperm(grid_h*grid_w)[:samples_per_frame]
    rows, cols = torch.unravel_index(flat_indices, (grid_h, grid_w))


    u,v = feature_to_pixel(rows,cols)
    stacked = torch.stack([u, v], dim=1)
    pixelcords_list.append(stacked)

    pose_list.append(pose)

    fancy_indexing = img_featured[:, rows, cols]
    fancy_indexing = fancy_indexing.permute(1,0)
    feature_list.append(fancy_indexing)



    frame_index = torch.full((samples_per_frame,), frame_idx)
    frameindices_list.append(frame_index)

all_poses = torch.stack(pose_list, dim=0)
all_features = torch.stack(feature_list, dim=0)
all_frameIndices = torch.stack(frameindices_list, dim=0)
all_pixelcoords = torch.stack(pixelcords_list, dim=0)

print("all_poses shape: ", all_poses.shape)
print("all_features shape: ", all_features.shape)
print("all_frameIndices shape: ", all_frameIndices.shape)
print("all_pixelcoords shape: ", all_pixelcoords.shape)
