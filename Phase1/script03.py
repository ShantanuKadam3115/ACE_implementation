import torch.nn as nn # type: ignore
import torch  # type: ignore


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,1)


    def forward(self, x):
        return self.layer1(x)
        


L1 = LinearModel()
input = torch.tensor([[10.0], [20.0], [30.0]])
output = L1(input)

print(output)

for params in L1.parameters():
    print(params)

    