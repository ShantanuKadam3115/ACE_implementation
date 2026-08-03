import torch  # type: ignore


fake_data = [(11, 31.5), (20.5, 52), (30, 69)]

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
w.requires_grad_()
b.requires_grad_()

for epoch in range(50):
    print("epoch: ", epoch)
    for i in range(len(fake_data)):
        x, y = fake_data[i]
        prediction = w * x + b 
        loss = (prediction - y) ** 2
        print("loss: ", loss.item())
        loss.backward()
        # print("w.grad: ", w.grad.item(), "b.grad: ", b.grad.item())
        with torch.no_grad():
            w -= 0.00001 * w.grad
            b -= 0.00001 * b.grad
        print("W: ", w.item(), "b: ", b.item())
        w.grad.zero_()
        b.grad.zero_()

