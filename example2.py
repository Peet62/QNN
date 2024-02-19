import torch
import numpy as np
import pandas as pd
import quaternion
import layers
import utils
import time
import matplotlib.pyplot as plt

cases = 250000
learning_rate = 0.1
epochs = 10

# input and output normalized data

train = pd.read_csv('input_data.csv', header=None)
target = pd.read_csv('output_data.csv', header=None)

x = torch.zeros(cases, 4, dtype=torch.float32, device='cuda')
x[:, 1:4] = torch.tensor(train.to_numpy())
x.requires_grad = True

# OUTPUTS
y = torch.zeros(cases, 4, dtype=torch.float32)
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

loss_function = torch.nn.MSELoss()

model = model.to('cuda')

# little software experimentation with L-BFGS hyperparameters is recommended...
# These settings lead to good results in specific regression task, lr can be reduced to 1e-2 -> 1e-3

lbfgs = torch.optim.LBFGS([x], history_size=10, max_iter=4, lr=learning_rate, tolerance_grad=1e-7, tolerance_change=1e-9,
                          line_search_fn='strong_wolfe')

history_lbfgs = []

losses = torch.zeros(epochs, 1, device='cuda')

start_time = time.time()

for epoch in range(epochs):
    def closure():
        if torch.is_grad_enabled():
            lbfgs.zero_grad()
        pred_y = model(x)
        loss = loss_function(pred_y, y)
        if loss.requires_grad:
            loss.backward()
        losses[epoch, 0] = loss
        return loss

    lbfgs.step(closure)

# torch.save(model.state_dict(), 'm1.pt')

losses2 = list(losses.detach().to('cpu'))

torch.save(losses2, 'losses.pt')

print(" --- %s seconds ---" % (time.time() - start_time))
plt.plot(losses2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f" % learning_rate)
plt.show()

print('Finished Training')