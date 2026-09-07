import torch  # type: ignore
from torch.utils.data import DataLoader, Dataset

#y= 2x +10

x = torch.arange(1, 22 + 1, dtype=torch.float32).reshape(-1, 1)

noise = torch.randn(22,1)*2

y = 2 *x + 10 + noise

# print(x, y)


class LinearDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = LinearDataset(x,y)

# print(dataset.__getitem__(5))

dataloader = DataLoader(dataset, 4, shuffle=True)



# print("dataset: ", type(dataset))
# print("dataloader: ", len(dataloader))

if __name__ == "__main__":
    # for batch_idx, (batch_x, batch_y) in enumerate(dataloader):

    #     # print(f"Batch {batch_idx + 1}")
    #     # print("x shape:", batch_x.shape)
    #     # print("y shape:", batch_y.shape)
    #     print(batch_x)
    #     print(batch_y)
    pass


