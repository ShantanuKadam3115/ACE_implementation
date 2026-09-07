import torch.nn as nn # type: ignore
import torch  # type: ignore
from torch.utils.data import DataLoader

#y = 2x +10

from script05 import x, y, LinearDataset, dataloader

# X = torch.tensor([[1.0],[2.0],[3.0]])
# Y = torch.tensor([[12.0],[14.0],[16.0]])

# dataset = LinearDataset(x,y)

# datLoader = DataLoader(dataset, batch_size=4, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1)

    def forward(self, x):
        return self.layer(x)

linearModel = Model()

lossFunction = nn.MSELoss()

optimizer = torch.optim.SGD(linearModel.parameters(),lr=0.001)



for epoch in range(1000):

    loss_sum = 0

    for batch_id, (x, y) in enumerate(dataloader):
    
        optimizer.zero_grad()

        prediction = linearModel(x)

        loss = lossFunction(prediction, y)

        # if epoch % 100 == 0:
        #     print(f"for {epoch} : Loss is {loss}")

        loss_sum = loss_sum + loss

        loss.backward()

        # print("w.grad:", linearModel.layer.weight.grad.item(), "b.grad:", linearModel.layer.bias.grad.item())

        optimizer.step()
    if epoch % 50 == 0:
        print(f"for the epoch: {epoch}, the average loss is {loss_sum/len(dataloader)}")



for params in linearModel.parameters():
    print(params)

