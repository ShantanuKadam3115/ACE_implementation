import torch #type: ignore
import torch.nn as nn #type: ignore
from torchvision.models import resnet18, ResNet18_Weights #type: ignore
import numpy as np
from torchvision import transforms #type: ignore

from Phase3.dataset import CustomDataset

datsetPath = "Phase3\\dataset\\seq-01\\seq-01"

test_dataset = CustomDataset(datsetPath)

img, pose = test_dataset[2]


img = img.float()/255
img.requires_grad = True

transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalized_img = transform(img)
batched_img  = normalized_img.unsqueeze(0)
# batched_img.requires_grad = True

model = resnet18(weights = ResNet18_Weights.DEFAULT)

truncate = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
truncate.eval()

for items in truncate.parameters():
    items.requires_grad = False
    # print(items.requires_grad)


result = truncate(batched_img )

output_value = result[0, :, 30, 40]
output_value.sum().backward()
mask = img.grad.abs().sum(dim=0) > 0

coordinates = torch.nonzero(mask)

min_row, max_row = coordinates[:, 0].min().item(), coordinates[:, 0].max().item()
min_col, max_col = coordinates[:, 1].min().item(), coordinates[:, 1].max().item()

print(f"({min_row}, {min_col}), ({min_row}, {max_col}), ({max_row}, {min_col}), ({max_row}, {max_col})")
# print(coordinates)


def feature_to_pixel(row_idx, col_idx, stride=4):
    u = col_idx * stride + stride / 2
    v = row_idx * stride + stride / 2
    return u, v


u,v = feature_to_pixel(30, 40)

print(u,v)
# print("result shape : ", result.shape)
# print(result.requires_grad)
# print(img)
# print(normalized_img)
# print("batched_img shape", batched_img.shape)
# print(ResNet18_Weights.DEFAULT.transforms())
# print(output_value, output_value.sum())



