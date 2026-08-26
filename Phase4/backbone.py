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

transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalized_img = transform(img)
batched_img  = normalized_img.unsqueeze(0)

model = resnet18(weights = ResNet18_Weights.DEFAULT)

truncate = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
truncate.eval()

for items in truncate.parameters():
    items.requires_grad = False
    # print(items.requires_grad)


result = truncate(batched_img )

print(result, result.shape)
print(result.requires_grad)
# print(img)
# print(normalized_img)
# print(resized_img.shape)
# print(ResNet18_Weights.DEFAULT.transforms())