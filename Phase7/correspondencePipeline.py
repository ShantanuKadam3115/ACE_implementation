from Phase4.backbone import TruncatedResNetBackbone , normalizedImage , feature_to_pixel
from Phase6.MLPCreation import MLP
from Phase3.dataset import CustomDataset
import torch  #type: ignore
import numpy as np


datasetPath = "Phase7\\testdata"
model = MLP()
model.load_state_dict(torch.load("Phase6/mlp_weights.pt"))
model.eval()

backbone = TruncatedResNetBackbone()
test_dataset = CustomDataset(datasetPath)
img, _ = test_dataset[0]
img_normalized = normalizedImage(img)

with torch.no_grad():
    img_featured = backbone(img_normalized)

img_featured = img_featured.squeeze(0)
_, grid_h, grid_w = img_featured.shape
print(grid_h,grid_w)

flat_indices = torch.arrange(grid_h*grid_w)
print( flat_indices.shape)
rows, cols = torch.unravel_index(flat_indices, (grid_h, grid_w))
print(rows.max())

sampled_features = img_featured[:, rows, cols]
sampled_features = sampled_features.permute(1,0)
print(sampled_features.shape)

u,v = feature_to_pixel(rows,cols)
pixels = torch.stack([u, v], dim=1)
print(pixels.shape)

with torch.no_grad():
    predicted_world_points = model(sampled_features)

print(predicted_world_points.shape)

# print(batchedImage.shape)
# print(img_featured.shape)
# print(model)


