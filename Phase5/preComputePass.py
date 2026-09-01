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

fancy_indexing = img_featured[:512,512]

# print(flat_indices.shape)
# print(rows.shape, cols.shape)
# print(rows, cols)
# print(img_featured.shape)
print(fancy_indexing)

