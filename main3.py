# THIS CODE REPLICATES MAIN.PY
# OPTIMIZER = ADAM / SGD
from prettytable import PrettyTable
import torch
import numpy as np
import pandas as pd
import quaternion
import layers
import utils
import time
import matplotlib.pyplot as plt


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


cases = 250000
epochy = 500
learning_rate = 0.01

train  = pd.read_csv('input_data.csv', header=None)
target = pd.read_csv('output_data.csv', header=None)

# 100000 batches of quaternions -> INPUTS

x = torch.zeros(cases, 4, dtype=torch.float32)  # quaternion.QuaternionTensor(torch.rand(cases, 4))
x[:, 1:4] = torch.tensor(train.to_numpy())

# b = # 0.001 * torch.rand(cases, 4)

# OUTPUTS
y = torch.zeros(cases, 4, dtype=torch.float32)  # quaternion.QuaternionTensor(torch.rand(cases, 4))
# x * torch.tensor([1.6]) + b

y[:, 1:4] = torch.tensor(target.to_numpy())
y[:, 1] = y[:, 1]
y[:, 2] = y[:, 3]

# x = x.to('cuda')
# y = y.to('cuda')

# x = x * torch.rand(2)  # Multiply with real-valued scalars
# x.norm().sum().backward()  # Take the absolute value, sum, and take the gradient

# Batch the data using a custom collate_fn function to convert to quaternion-valued images

model = torch.nn.Sequential(
    layers.QLinear(1, 2, bias=True),
    torch.nn.ReLU(),
    layers.QLinear(2, 1, bias=True),
    torch.nn.ReLU(),
    # layers.QuaternionToReal(1),  # Take the absolute value in output
)
# model = model.to('cuda')

# 'originally'  criterion = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = torch.zeros(epochy, 1, device='cpu')

start_time = time.time()

for epoch in range(epochy):
    # zero the parameter gradients

    optimizer.zero_grad()
    pred_y = model(x)
    # real component does not matter...
    if epoch == (epochy-1):
        print(pred_y[0:16, :] - y[0:16, :])

    loss = loss_function(pred_y, y)
    # losses.append(loss.item())
    losses[epoch, 0] = loss
    # forward + backward + optimize
    loss.backward()
    optimizer.step()

losses2 = list(losses.detach().to('cpu'))

# torch.save(model.state_dict(), 'a1.pt')

torch.save(losses2, 'losses_a5b.pt')

count_parameters(model)

print(" --- %s seconds ---" % (time.time() - start_time))
plt.plot(losses2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f" % learning_rate)
plt.show()

print('Finished Training')