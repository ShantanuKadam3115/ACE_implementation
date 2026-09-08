import torch #type: ignore
from torch.utils.data import Dataset #type: ignore




class Bufferdataset(Dataset):
    def __init__(self, path):
        self.directory = path
        self.data = torch.load(self.directory)
        
    def __len__(self):
        return self.data["features"].shape[0]

    def __getitem__(self, idx):
        feature = self.data["features"][idx]
        pixel_coords = self.data["pixel_coords"][idx]
        index = self.data["frame_indices"][idx]
        pose = self.data["poses"][index]
        return feature, pixel_coords, pose

if __name__ == "__main__":
    bufferPath = "seq_1_buffer.pt"
    data = Bufferdataset(path=bufferPath)
    feature, pixel_coords, pose = data[0]
    print(feature.shape, pixel_coords, pose)