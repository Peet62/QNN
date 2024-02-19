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
epochs = 500
learning_rate = 0.01

train = pd.read_csv('input_data.csv', header=None)
target = pd.read_csv('output_data.csv', header=None)

x = torch.zeros(cases, 4, dtype=torch.float32, device='cuda')
x[:, 1:4] = torch.tensor(train.to_numpy())
x.requires_grad = True

# OUTPUTS
y = torch.zeros(cases, 4, dtype=torch.float32)  # quaternion.QuaternionTensor(torch.rand(cases, 4))
z = torch.zeros(cases, 4, dtype=torch.float32)

z[:, 1:4] = torch.tensor(target.to_numpy())

# only 2 quaternion bases are used (i, j)
y[:, 1] = z[:, 1]
y[:, 2] = z[:, 3]

y = y.to('cuda')

# nn.ReLU can be replaced with nn.GELU, nn.Tanh or completely omitted.
# if AF is omitted, the activation function can be considered as quaternion linear AF, the quaternion variant of Matlab "purelin"
# bias parameter was default set to True, if omitted, influence on MSE performance, but number of parameters significantly decrease in small QNN

model = torch.nn.Sequential(
    layers.QLinear(1, 2, bias=True),
    # torch.nn.ReLU(),
    layers.QLinear(2, 1, bias=True),
    # torch.nn.ReLU(),
    # layers.QuaternionToReal(1),  # Take the absolute value in output
)
model = model.to('cuda')

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = torch.zeros(epochs, 1, device='cuda')

start_time = time.time()

for epoch in range(epochs):
    # zero the parameter gradients

    optimizer.zero_grad()
    pred_y = model(x)
    # real component does not matter...
    loss = loss_function(pred_y, y)
    losses[epoch, 0] = loss
    # forward + backward + optimize
    loss.backward()
    optimizer.step()

losses2 = list(losses.detach().to('cpu'))

# torch.save(model.state_dict(), 'a1.pt')

torch.save(losses2, 'losses.pt')

count_parameters(model)

print(" --- %s seconds ---" % (time.time() - start_time))
plt.plot(losses2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f" % learning_rate)
plt.show()

print('Finished Training')