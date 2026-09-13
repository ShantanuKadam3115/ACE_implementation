import torch #type: ignore
import torch.nn as nn  #type: ignore
import torch.nn.functional as F  #type: ignore
from torch.utils.data import TensorDataset, DataLoader #type: ignore

data = torch.load("seq_1_buffer.pt")

features = data["features"]
dataset = TensorDataset(features)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


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

# first_batch_predictions = torch.empty(2000 , 3)
# print(len(loader))
# first_batch = next(iter(loader))
# x_batch = first_batch[0]   # unpack the tuple from TensorDataset
# print(x_batch.shape)       # (256, 64)
# print(x_batch[0])

if __name__ == "__main__":

    for (x_batch) in loader:
        for i in range(len(x_batch)):
            

            predicted = linearModel(x_batch[i])
            result.append(predicted)

            
        result = torch.cat(result, dim=0)
        break


    print(result.shape)

