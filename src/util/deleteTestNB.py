import torch
from mydistributions import *
import numpy as np
from scipy.stats import nbinom

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.zdim = outputSize
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.phi = torch.nn.Parameter(torch.tensor([torch.rand(1)]), requires_grad=True)

    def forward(self, x):
        out = self.linear(x)
        return NegativeBinomial(torch.exp(out), self.phi)

x = torch.rand(1000,2)
xnp = x.numpy()
mu = np.exp(1 + xnp.dot([3., -2.1]))

phi = 0.14

var = mu + ((mu ** 2) / phi)

p = mu / var
n = (mu**2) / (var - mu)

y = torch.tensor(nbinom.rvs(n,p), dtype=float)

#
xt = torch.rand(100,2)
xtnp = xt.numpy()
mut = np.exp(1 + xtnp.dot([3., -2.1]))

phit = phi

vart = mut + ((mut ** 2) / phit)

pt = mut / vart
nt = (mut**2) / (vart - mut)

yt = torch.tensor(nbinom.rvs(nt,pt), dtype=float)



model = LinearRegression(2,1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(x)

    # get loss for the predicted output
    loss = -torch.mean(outputs.log_prob(y))
    #print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, training loss {}'.format(epoch, loss.item()))

    with torch.no_grad():
        outputs = model(xt)

        # get loss for the predicted output
        loss = -torch.mean(outputs.log_prob(yt))
        print('validation loss %.3f' % loss)

    print('gene-wise disp: %.3f' % model.phi)
