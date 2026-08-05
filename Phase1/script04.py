import torch.nn as nn # type: ignore
import torch  # type: ignore

#y = 2x +10


X = torch.tensor([[1.0],[2.0],[3.0]])
Y = torch.tensor([[12.0],[14.0],[16.0]])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1)

    def forward(self, x):
        return self.layer(x)

linearModel = Model()

lossFunction = nn.MSELoss()

optimizer = torch.optim.SGD(linearModel.parameters(),lr=0.01)



for epoch in range(2000):
    prediction = linearModel(X)

    loss = lossFunction(prediction,Y)

    if epoch % 100 == 0:
        print(f"for {epoch} : Loss is {loss}")

    optimizer.zero_grad()

    loss.backward()

    # print("w.grad:", linearModel.layer.weight.grad.item(), "b.grad:", linearModel.layer.bias.grad.item())

    optimizer.step()



for params in linearModel.parameters():
    print(params)

