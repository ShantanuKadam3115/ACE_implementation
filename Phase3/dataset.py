import torch.nn as nn # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset #type: ignore
import glob
import os
import numpy as np
from PIL import Image


datsetPath = "Phase3\\dataset\\seq-01\\seq-01"


fx, fy  = 525.0, 525.0
cx, cy = 319.5,239.5
k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])

class CustomDataset(Dataset):
    def __init__(self,path):
        self.seq_dir = path
        image_files_path = os.path.join(self.seq_dir, "*.color.png")
        self.num_frames = len(glob.glob(image_files_path))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        padded = f"{idx:06d}"
        image_path = os.path.join(self.seq_dir, f"frame-{padded}.color.png")
        txt_path = os.path.join(self.seq_dir, f"frame-{padded}.pose.txt")

        image_tensor = torch.from_numpy(np.array(Image.open(image_path))).permute(2,0,1)

        pose_tensor = torch.from_numpy(np.loadtxt(txt_path).astype(np.float32))

        return (image_tensor, pose_tensor) 
        
if __name__ == "__main__":
    dataset = CustomDataset(datsetPath)
    img, pose = dataset[10]
    print(img.shape, img.dtype)
    print(pose, pose.dtype)

