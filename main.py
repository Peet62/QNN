import torch
import numpy as np
import pandas as pd
import quaternion
import layers
import utils
import time
import matplotlib.pyplot as plt

cases = 250000
epochy = 100
learning_rate = 0.01

train = pd.read_csv('input_data.csv', header=None)
target = pd.read_csv('output_data.csv', header=None)

# 100000 batches of quaternions -> INPUTS

x = torch.zeros(cases, 4, dtype=torch.float32, device='cuda')  # quaternion.QuaternionTensor(torch.rand(cases, 4))
x[:, 1:4] = torch.tensor(train.to_numpy())
x.requires_grad = True

# b = # 0.001 * torch.rand(cases, 4)

# OUTPUTS
y = torch.zeros(cases, 4, dtype=torch.float32)  # quaternion.QuaternionTensor(torch.rand(cases, 4))
z = torch.zeros(cases, 4, dtype=torch.float32)  # quaternion.QuaternionTensor(torch.rand(cases, 4))
# x * torch.tensor([1.6]) + b

z[:, 1:4] = torch.tensor(target.to_numpy())
# chcem 1, 3 originalne...
y[:, 1] = z[:, 1]
y[:, 2] = z[:, 3]

# y[:, 0] = y[:, 1]
# y[:, 2] = y[:, 3]

# x = x.to('cuda')
y = y.to('cuda')

# x = x * torch.rand(2)  # Multiply with real-valued scalars
# x.norm().sum().backward()  # Take the absolute value, sum, and take the gradient

# Batch the data using a custom collate_fn function to convert to quaternion-valued images

model = torch.nn.Sequential(
    layers.QLinear(1, 2, bias=True),
    torch.nn.ReLU(),  # Tanh(),
    layers.QLinear(2, 1, bias=True),
    # torch.nn.ReLU(),
    # layers.QuaternionToReal(1),  # Take the absolute value in output
)

loss_function = torch.nn.MSELoss()

# def closure():
#     if torch.is_grad_enabled():
#        lbfgs.zero_grad()
#    pred_y = model(x)
#    loss = loss_function(pred_y, y)
#    if loss.requires_grad:
#        loss.backward()
#    return loss


model = model.to('cuda')

# 'originally'  criterion = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lbfgs = torch.optim.LBFGS([x], history_size=10, max_iter=4, lr=0.1, tolerance_grad=1e-7, tolerance_change=1e-9,
                          line_search_fn='strong_wolfe')

history_lbfgs = []

losses = torch.zeros(epochy, 1, device='cuda')

start_time = time.time()

for epoch in range(epochy):
    def closure():
        if torch.is_grad_enabled():
            lbfgs.zero_grad()
        pred_y = model(x)
        if epoch == epochy - 1:
            print(pred_y[0:16, :] - y[0:16, :])
        loss = loss_function(pred_y, y)
        if loss.requires_grad:
            loss.backward()
        losses[epoch, 0] = loss
        return loss


    lbfgs.step(closure)

# torch.save(model.state_dict(), 'm1.pt')

losses2 = list(losses.detach().to('cpu'))

torch.save(losses2, 'losses_q12b.pt')

print(" --- %s seconds ---" % (time.time() - start_time))
plt.plot(losses2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f" % learning_rate)
plt.show()

print('Finished Training')
