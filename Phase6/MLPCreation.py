import torch #type: ignore
import torch.nn as nn  #type: ignore
import torch.nn.functional as F  #type: ignore
from torch.utils.data import TensorDataset, DataLoader #type: ignore

data = torch.load("seq_1_buffer.pt")

features = data["features"]
pose = data["poses"]
pixel_coords = data["pixel_coords"]

feature_dataset = TensorDataset(features)
feature_loader = DataLoader(feature_dataset, batch_size=2048, shuffle=False)

pose_dataset = TensorDataset(pose)
pose_loader = DataLoader(pose_dataset, batch_size = 4, shuffle=False )

fx, fy  = 532.57, 531.54
cx, cy = 320.0,240.0
k = torch.tensor([[fx,0.0, cx],[0.0,fy,cy],[0.0,0.0,1.0]])

R_cw = pose[0:3, 0:3]
t_cw = pose[0:3, 3]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 30)
        self.layer2 = nn.Linear(30, 24)
        self.layer3 = nn.Linear(24, 12)
        self.layer4 = nn.Linear(12, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

linearModel = MLP()

lossFunction = nn.MSELoss()

optimizer = torch.optim.SGD(linearModel.parameters(),lr=0.001)

result = []
predicted_world_cords = []

# first_batch_predictions = torch.empty(2000 , 3)
# print(len(feature_loader))
# first_batch = next(iter(feature_loader))
# x_batch = first_batch[0]   # unpack the tuple from TensorDataset
# print(x_batch.shape)       # (250, 64)
# print(x_batch[0].shape)

# print(len(pose_loader))
first_batch = next(iter(pose_loader))
pose_batch = first_batch[0]   # unpack the tuple from TensorDataset
# print(y_batch.shape)       # (250, 64)
# print(y_batch[0].shape)



if __name__ == "__main__":

    for (x_batch) in feature_loader:
        for i in range(len(x_batch)):
            
            predicted = linearModel(x_batch[i])
            result.append(predicted)
           
        result = torch.cat(result, dim=0)
        break

    feature_chunks = torch.split(result,512)

    for feature_chunks, pose_batch in zip(feature_chunks,pose_batch):
        R_cw = pose_batch[0:3, 0:3]
        t_cw = pose_batch[0:3, 3]

        R_wc = R_cw.mT
        t_wc = - R_wc @ t_cw.T #[0,0,5]
        
        X_cam = feature_chunks@ R_wc.T + t_wc
        u1,v1, w1 = k @ X_cam.T
        u1 = u1/w1
        v1 = v1/w1
        uv = torch.stack([u1, v1], dim=1)   # (512, 2)
        predicted_world_cords.append(uv)
    predicted_world_cords = torch.cat(predicted_world_cords, dim =0)

    pixel_coords_batch1 = pixel_coords[:2048]
    

    Loss = lossFunction(predicted_world_cords, pixel_coords_batch1)

    print(Loss)


            


    

    



