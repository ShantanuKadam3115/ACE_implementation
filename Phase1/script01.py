import torch  # type: ignore
import numpy as np

x_data = torch.tensor([4,5,6])

print(x_data.shape, x_data.dtype)

data = [1,2,3]
numpy_data = np.array(data)
x_np_data = torch.from_numpy(numpy_data)

print(x_np_data.dtype)

random_tensor = torch.rand(3)

print(random_tensor)


tensor_operation = x_np_data * random_tensor

print(tensor_operation)



