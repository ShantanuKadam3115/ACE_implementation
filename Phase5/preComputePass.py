from Phase4.backbone import TruncatedResNetBackbone, feature_to_pixel, normalizedImage
from Phase3.dataset import CustomDataset
import torch #type: ignore

datasetPath = "Phase3\\dataset\\seq-01\\seq-01"
dataset = CustomDataset(datasetPath)
backbone = TruncatedResNetBackbone()
samples_per_frame = 512

img, pose = dataset[0]

img_normalized = normalizedImage(img)

with torch.no_grad():
    img_featured = backbone(img_normalized)
img_featured = img_featured.squeeze(0)
_, grid_h, grid_w = img_featured.shape


flat_indices = torch.randperm(grid_h*grid_w)[:samples_per_frame]

rows, cols = torch.unravel_index(flat_indices, (grid_h, grid_w))



fancy_indexing = img_featured[:, rows, cols]
fancy_indexing = fancy_indexing.permute(1,0)


u,v = feature_to_pixel(rows,cols)

frame_index = torch.full((samples_per_frame,), 0)

# print(u,v)
print(frame_index.shape)
# print(img_featured[:,rows[0],cols[0]])
# print(flat_indices.shape)
# print(rows.shape, cols.shape)
# print(rows, cols)
# print(img_featured.shape)

# print(fancy_indexing)

# print(fancy_indexing.shape)

